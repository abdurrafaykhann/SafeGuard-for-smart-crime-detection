"""Build a clean YOLO dataset (gun / knife) from an Ultralytics NDJSON export, plus your own negatives.

    python scripts/weapons/prepare_dataset.py --ndjson weapon-dataset-main-yolo.ndjson --out data/weapons \
        --negatives instance/weapons_site/negatives --max-train 12000

What it does, and why:
  * keeps only `gun` and `knife`. The dataset's generic `weapon` and `stick-rod` boxes are ambiguous, so any
    image containing one is dropped rather than half-labelled; `person` boxes are ignored (unlabelled
    people are simply background for this detector);
  * re-splits by SOURCE PHOTO (Roboflow exports contain several augmented copies of one photo; the
    original split put ~1,800 such photos on both sides of train/test, which inflates any test score);
  * downloads the images itself from the NDJSON links (they are signed CDN URLs that expire, see
    the `Expires=` value) and verifies every file decodes;
  * adds your webcam negatives as images with empty label files, so empty desks / hands / look-alike
    objects are explicitly taught as "nothing here".
Held-out webcam sets (negatives_test, knife_test) are NEVER added here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

KEEP = {"gun": 0, "knife": 1}
DROP_IF_PRESENT = {"weapon", "stick-rod"}  # ambiguous classes: skip the whole image
MIN_SIDE = 0.01  # ignore boxes thinner than 1% of the image


def source_key(filename: str) -> str:
    m = re.match(r"(.+?)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+\.\w+$", filename, re.I)
    return m.group(1) if m else re.sub(r"\.\w+$", "", filename)


def group_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


def load(ndjson: Path):
    rows = [json.loads(line) for line in ndjson.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = rows[0]["class_names"]
    if isinstance(names, str):
        import ast
        names = ast.literal_eval(names)
    return {int(k): v for k, v in names.items()}, rows[1:]


def boxes_of(rec: dict) -> list:
    a = rec["annotations"]
    if isinstance(a, str):
        import ast
        a = ast.literal_eval(a)
    return a.get("boxes", [])


def fetch(url: str, retries: int = 3) -> bytes | None:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception:
            time.sleep(1 + i)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negatives", type=Path, help="folder of your own webcam frames with no weapon in view")
    ap.add_argument("--max-train", type=int, default=12000, help="cap on training images (keeps Colab runs short)")
    ap.add_argument("--limit-total", type=int, help="smoke test: stop after this many dataset images")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    names, imgs = load(a.ndjson)
    keep_ids = {i for i, n in names.items() if n in KEEP}
    drop_ids = {i for i, n in names.items() if n in DROP_IF_PRESENT}
    plan, stats = [], Counter()
    for r in imgs:
        bs = boxes_of(r)
        ids = {int(b[0]) for b in bs}
        if ids & drop_ids:
            stats["dropped: has ambiguous weapon/stick-rod box"] += 1
            continue
        lines = []
        for c, x, y, w, h in bs:
            if int(c) in keep_ids and float(w) >= MIN_SIDE and float(h) >= MIN_SIDE:
                lines.append(f"{KEEP[names[int(c)]]} {float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}")
        if not lines:
            stats["dropped: no usable gun/knife box"] += 1
            continue
        plan.append((r, lines, group_split(source_key(r["file"]))))
    by_split = defaultdict(list)
    for item in plan:
        by_split[item[2]].append(item)
    # cap training images by whole source groups so copies of one photo stay together
    if len(by_split["train"]) > a.max_train:
        groups = defaultdict(list)
        for it in by_split["train"]:
            groups[source_key(it[0]["file"])].append(it)
        keys = list(groups)
        random.shuffle(keys)
        kept, n = [], 0
        for k in keys:
            if n >= a.max_train:
                break
            kept += groups[k]; n += len(groups[k])
        stats["train images skipped by --max-train"] = len(by_split["train"]) - len(kept)
        by_split["train"] = kept
    if a.limit_total:
        for s in list(by_split):
            by_split[s] = by_split[s][: max(1, a.limit_total // 3 if s == "train" else a.limit_total // 6)]

    for s in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (a.out / kind / s).mkdir(parents=True, exist_ok=True)

    def work(item):
        rec, lines, split = item
        data = fetch(rec["url"])
        if data is None:
            return rec["file"], split, "download failed"
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return rec["file"], split, "not a valid image"
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(rec["file"]).stem)[:90]
        (a.out / "images" / split / f"{stem}.jpg").write_bytes(data if rec["file"].lower().endswith((".jpg", ".jpeg")) else cv2.imencode(".jpg", im)[1].tobytes())
        (a.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        return rec["file"], split, None

    items = [it for s in ("train", "val", "test") for it in by_split[s]]
    print(f"downloading {len(items)} images with {a.workers} workers "
          f"(train {len(by_split['train'])}, val {len(by_split['val'])}, test {len(by_split['test'])})...", flush=True)
    ok, failures = Counter(), []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for n, f in enumerate(as_completed(futs), 1):
            fn, split, err = f.result()
            if err:
                failures.append((fn, err))
            else:
                ok[split] += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(items)}", flush=True)

    neg_counts = Counter()
    if a.negatives and a.negatives.exists():
        files = sorted(p for p in a.negatives.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        random.shuffle(files)
        n_val = max(1, len(files) // 7)
        for i, p in enumerate(files):
            split = "val" if i < n_val else "train"
            dst = a.out / "images" / split / f"neg_{p.stem}.jpg"
            shutil.copyfile(p, dst)
            (a.out / "labels" / split / f"neg_{p.stem}.txt").write_text("")  # empty = no objects
            neg_counts[split] += 1

    (a.out / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: gun\n  1: knife\n")
    manifest = {"source": a.ndjson.name, "classes": ["gun", "knife"], "downloaded": dict(ok), "failed": len(failures),
                "negatives": dict(neg_counts), "filtering": dict(stats)}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if failures:
        print("first failures:", failures[:3])
    return 0 if len(failures) < 0.02 * max(len(items), 1) else 1


if __name__ == "__main__":
    sys.exit(main())

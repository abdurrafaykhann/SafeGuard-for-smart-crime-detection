"""Build a SMALL gun/knife YOLO dataset (CPU-trainable) from the Ultralytics NDJSON export, biased towards close-up knives.

    python scripts/weapons/make_knife_subset.py NDJSON OUT_DIR [--close 700 --knife 400 --gun 500 --neg 250 --webcam-neg 200]

Classes: 0 gun, 1 knife. Images that contain the dataset's vague 'weapon' or 'stick-rod' boxes are skipped (they would teach the
model that real weapons are background). A knife counts as "close-up" when its box covers at least 4% of the picture. Validation
images come from the export's own val split and are only used to pick the best epoch. The held-out webcam frames in
instance/weapons_site/knife_test and negatives_test are NEVER added: they are the test set. Webcam 'negatives' (no weapon, your room)
are added as background examples only. Data licence of the export: not stated; this is for local training and evaluation."""
from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def boxes_of(row):
    a = row["annotations"]
    a = ast.literal_eval(a) if isinstance(a, str) else a
    return a.get("boxes", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ndjson", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--close", type=int, default=700)
    ap.add_argument("--knife", type=int, default=400)
    ap.add_argument("--gun", type=int, default=500)
    ap.add_argument("--neg", type=int, default=250)
    ap.add_argument("--webcam-neg", type=int, default=200)
    a = ap.parse_args()
    rng = random.Random(1)
    rows = [json.loads(l) for l in a.ndjson.read_text(encoding="utf-8").splitlines()]
    meta, imgs = rows[0], rows[1:]
    cn = meta["class_names"]
    cn = ast.literal_eval(cn) if isinstance(cn, str) else cn
    name = {int(k): v for k, v in cn.items()}
    ids = {v: k for k, v in name.items()}

    def classify(r):
        b = boxes_of(r)
        cls = {int(x[0]) for x in b}
        if cls & {ids["stick-rod"], ids["weapon"]}:
            return None
        knives = [x for x in b if int(x[0]) == ids["knife"]]
        guns = [x for x in b if int(x[0]) == ids["gun"]]
        if knives and any(x[3] * x[4] >= 0.04 for x in knives):
            return "close"
        if knives:
            return "knife"
        if guns:
            return "gun"
        return "neg" if b else None

    pools = {"train": {"close": [], "knife": [], "gun": [], "neg": []}, "val": {"close": [], "knife": [], "gun": [], "neg": []}}
    for r in imgs:
        sp = r.get("split")
        k = classify(r)
        if sp in pools and k:
            pools[sp][k].append(r)
    want = {"train": {"close": a.close, "knife": a.knife, "gun": a.gun, "neg": a.neg}, "val": {"close": 100, "knife": 50, "gun": 100, "neg": 50}}
    chosen = []
    for sp in pools:
        for k, n in want[sp].items():
            pool = pools[sp][k]
            rng.shuffle(pool)
            chosen += [(sp, k, r) for r in pool[:n]]
            print(f"{sp:5s} {k:6s}: {min(n, len(pool))} of {len(pool)} available", flush=True)

    if a.out.exists():
        shutil.rmtree(a.out)
    for sp in ("train", "val"):
        (a.out / "images" / sp).mkdir(parents=True)
        (a.out / "labels" / sp).mkdir(parents=True)

    def fetch(item):
        sp, k, r = item
        stem = Path(r["file"]).stem[:80].replace(" ", "_")
        dest = a.out / "images" / sp / f"{stem}.jpg"
        for _ in range(3):
            try:
                req = urllib.request.Request(r["url"], headers={"User-Agent": "Mozilla/5.0"})
                dest.write_bytes(urllib.request.urlopen(req, timeout=40).read())
                lines = []
                for x in boxes_of(r):
                    c = int(x[0])
                    if c in (ids["gun"], ids["knife"]):
                        lines.append(f"{0 if c == ids['gun'] else 1} {x[1]:.6f} {x[2]:.6f} {x[3]:.6f} {x[4]:.6f}")
                (a.out / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                return True
            except Exception:
                continue
        return False

    with ThreadPoolExecutor(24) as ex:
        ok = list(ex.map(fetch, chosen))
    print(f"downloaded {sum(ok)} of {len(chosen)} images", flush=True)

    neg_dir = ROOT / "instance" / "weapons_site" / "negatives"
    extra = sorted(neg_dir.glob("*.jpg"))
    rng.shuffle(extra)
    for i, f in enumerate(extra[: a.webcam_neg]):
        shutil.copy(f, a.out / "images" / "train" / f"webcamneg_{i:04d}.jpg")
        (a.out / "labels" / "train" / f"webcamneg_{i:04d}.txt").write_text("", encoding="utf-8")
    (a.out / "data.yaml").write_text(f"path: {a.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: gun\n  1: knife\n", encoding="utf-8")
    print("train images:", len(list((a.out / 'images' / 'train').glob('*.jpg'))), "| val images:", len(list((a.out / 'images' / 'val').glob('*.jpg'))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does face matching treat groups differently? Per-group stranger false-match and recognition rates, plus image-quality robustness.

    python scripts/evaluate_face_groups.py LFW_DIR ATTRIBUTES.txt [--out docs/evaluation_face_groups.json]

Uses the cached LFW embeddings from evaluate_gallery_scale.py. Groups come from the public LFW attribute file (Kumar et al.), whose
labels are themselves ESTIMATED by a classifier, not self-reported: treat the group results as a screening test, not proof of fairness.
For each group, with 1,000 people enrolled (one photo each, 5 random draws):
  * stranger false-match rate: strangers of that group who score above the threshold against someone in the gallery
  * recognition rate: enrolled people of that group accepted as the correct person
Robustness: the same test with query photos darkened, blurred or shrunk (gallery photos stay clean)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.faces import FaceEngine  # noqa: E402

THR = [0.45, 0.50]
N_GALLERY = 1000


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(c - h, 0), 5), round(min(c + h, 1), 5)]


def load_attributes(path: Path) -> dict[tuple[str, int], dict]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = lines[1].lstrip("#").strip().split("\t")
    out = {}
    for ln in lines[2:]:
        f = ln.split("\t")
        if len(f) < len(header):
            continue
        row = dict(zip(header, f))
        out[(row["person"].replace(" ", "_"), int(row["imagenum"]))] = {k: float(v) for k, v in row.items() if k not in ("person", "imagenum")}
    return out


def groups_for(attr: dict) -> dict[str, str]:
    race = {k: attr[k] for k in ("Asian", "White", "Black", "Indian") if k in attr}
    top = max(race, key=race.get)
    return {"gender": "male" if attr["Male"] > 0 else "female",
            "appearance": top.lower() if race[top] > 0 else "unclear",
            "age": "child" if attr.get("Child", -9) > 0 else ("senior" if attr.get("Senior", -9) > 0 else "adult")}


def evaluate(X, ids, qmask, pool_X=None, label_groups=None, repeats=5):
    """Return per-threshold, per-group counts. Queries are rows where qmask is True; pool_X (optional) replaces query embeddings."""
    people = np.array(sorted(set(ids)))
    first = {p: int(np.where(ids == p)[0][0]) for p in people}
    res = {t: {} for t in THR}
    for rep in range(repeats):
        rng = np.random.default_rng(rep)
        chosen = rng.permutation(len(people))[:N_GALLERY]
        enrolled = set(people[chosen])
        gal_rows = np.array([first[p] for p in people[chosen]])
        G, gal_ids = X[gal_rows], ids[gal_rows]
        gal_set = set(gal_rows.tolist())
        for i in np.where(qmask)[0]:
            if i in gal_set:
                continue
            q = (pool_X if pool_X is not None else X)[i]
            sims = G @ q
            j = int(np.argmax(sims))
            s, hit = float(sims[j]), gal_ids[j]
            is_enrolled = ids[i] in enrolled
            for grp in label_groups[i]:
                for t in THR:
                    c = res[t].setdefault(grp, {"enrolled": 0, "right": 0, "strangers": 0, "false_match": 0})
                    if is_enrolled:
                        c["enrolled"] += 1
                        c["right"] += (s >= t) and hit == ids[i]
                    else:
                        c["strangers"] += 1
                        c["false_match"] += s >= t
    return res


def summarise(res) -> dict:
    out = {}
    for t, groups in res.items():
        out[str(t)] = {g: {"enrolled_queries": c["enrolled"], "recognised": round(c["right"] / max(c["enrolled"], 1), 4),
                           "stranger_queries": c["strangers"], "stranger_false_match": round(c["false_match"] / max(c["strangers"], 1), 5),
                           "stranger_ci95": wilson(c["false_match"], c["strangers"])} for g, c in sorted(groups.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lfw", type=Path)
    ap.add_argument("attributes", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = np.load(a.lfw.parent / "embeddings_min2.npz", allow_pickle=False)
    X, ids = d["X"], d["ids"]
    attrs = load_attributes(a.attributes)
    counter: dict[str, int] = {}
    labels, keep = [], np.zeros(len(ids), bool)
    for i, p in enumerate(ids):
        counter[p] = counter.get(p, 0) + 1
        at = attrs.get((p, counter[p]))
        if at is None:
            labels.append([])
            continue
        g = groups_for(at)
        labels.append([f"gender:{g['gender']}", f"appearance:{g['appearance']}", f"age:{g['age']}", "all"])
        keep[i] = True
    print(f"{int(keep.sum())} of {len(ids)} faces have attribute labels", flush=True)
    out = {"note": "Group labels are estimated by a classifier (LFW attributes), not self-reported. Screening test only.", "gallery_people": N_GALLERY,
           "labelled_faces": int(keep.sum())}
    out["groups"] = summarise(evaluate(X, ids, keep, label_groups=labels))
    for t in ("0.5",):
        print(f"--- threshold {t}: group | recognised | stranger false-match (95% CI) | n strangers")
        for g, v in out["groups"][t].items():
            print(f"{g:22s} {v['recognised']:.1%}   {v['stranger_false_match']:.2%} ({v['stranger_ci95'][0]:.2%}-{v['stranger_ci95'][1]:.2%})   {v['stranger_queries']}", flush=True)

    # robustness: degrade the QUERY photos of a random subset, keep gallery photos clean
    eng = FaceEngine(Config.MODELS_DIR, Config.FACE_MATCH_THRESHOLD)
    rng = np.random.default_rng(7)
    subset = np.sort(rng.choice(len(ids), 2500, replace=False))
    files = {}
    cnt: dict[str, int] = {}
    for i, p in enumerate(ids):
        cnt[p] = cnt.get(p, 0) + 1
        if i in set(subset.tolist()):
            files[i] = a.lfw / p / f"{p}_{cnt[p]:04d}.jpg"
    degr = {"clean": lambda im: im, "dark (x0.35)": lambda im: np.clip(im.astype(np.float32) * 0.35, 0, 255).astype(np.uint8),
            "blur (sigma 3)": lambda im: cv2.GaussianBlur(im, (0, 0), 3), "low-res (48 px)": lambda im: cv2.resize(cv2.resize(im, (48, 48), interpolation=cv2.INTER_AREA), (250, 250))}
    out["robustness"] = {}
    all_labels = [["all"] for _ in ids]
    for name, fn in degr.items():
        pool = X.copy()
        ok = np.zeros(len(ids), bool)
        for i, f in files.items():
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = fn(img)
            face = eng.largest_face(img)
            e = eng.embed(img, face) if face is not None else None
            if e is not None:
                pool[i] = e
                ok[i] = True
        found = int(ok.sum())
        r = summarise(evaluate(X, ids, ok, pool_X=pool, label_groups=all_labels, repeats=3))
        out["robustness"][name] = {"faces_found": found, "faces_total": len(files), "results": {t: r[t]["all"] for t in r}}
        v = r["0.5"]["all"]
        print(f"{name:18s} face found {found}/{len(files)} | at 0.50: recognised {v['recognised']:.1%}, strangers matched {v['stranger_false_match']:.2%}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

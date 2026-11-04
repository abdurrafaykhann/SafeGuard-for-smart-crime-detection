"""Open-set face identification: how often is a STRANGER wrongly matched to someone in the gallery?

    python scripts/evaluate_open_set.py DATASET_DIR [--splits 30] [--out docs/evaluation_open_set.json]

evaluate_identification.py only asks "if the person is enrolled, is the right one found?". In surveillance the worse
error is the reverse: a person who is NOT enrolled scoring above the threshold and being shown with someone's case file.
Here the identities of a folder-per-person dataset are split at random into an enrolled half (the gallery) and a stranger
half that is never enrolled. For many random splits we measure, per threshold:
  * stranger false-match rate: a stranger photo whose best gallery similarity reaches the threshold
  * enrolled people: correct match / wrong person above threshold / rejected (below threshold)
Enrolled queries are leave-one-photo-out (the query photo is removed from the gallery)."""
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

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def embed_dataset(root: Path, eng: FaceEngine) -> tuple[np.ndarray, np.ndarray, int]:
    vecs, ids, skipped = [], [], 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir()):
            img = cv2.imread(str(f)) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else None
            face = eng.largest_face(img) if img is not None else None
            e = eng.embed(img, face) if face is not None else None
            if e is None:
                skipped += 1
                continue
            vecs.append(e)
            ids.append(folder.name)
    return np.stack(vecs), np.array(ids), skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    eng = FaceEngine(Config.MODELS_DIR, Config.FACE_MATCH_THRESHOLD)
    X, ids, skipped = embed_dataset(a.dataset, eng)
    people = sorted(set(ids))
    S = X @ X.T
    rng = np.random.default_rng(0)
    gen = {t: [0, 0, 0] for t in THRESHOLDS}   # correct, wrong-person, rejected
    strangers = {t: 0 for t in THRESHOLDS}
    n_gen = n_str = 0
    for _ in range(a.splits):
        order = rng.permutation(len(people))
        gallery_people = {people[i] for i in order[: len(people) // 2]}
        in_gallery = np.array([p in gallery_people for p in ids])
        for i in range(len(ids)):
            rows = np.where(in_gallery)[0]
            if in_gallery[i]:
                rows = rows[rows != i]
                if not (ids[rows] == ids[i]).any():
                    continue  # nobody left of this person to find
            sims = S[i, rows]
            best = int(np.argmax(sims))
            top_sim, top_id = float(sims[best]), ids[rows[best]]
            if in_gallery[i]:
                n_gen += 1
                for t in THRESHOLDS:
                    if top_sim < t:
                        gen[t][2] += 1
                    elif top_id == ids[i]:
                        gen[t][0] += 1
                    else:
                        gen[t][1] += 1
            else:
                n_str += 1
                for t in THRESHOLDS:
                    strangers[t] += top_sim >= t
    rows_out = []
    print(f"{len(people)} people, {len(ids)} photos ({skipped} skipped), {a.splits} random splits (half enrolled, half strangers)")
    print(f"enrolled queries {n_gen}, stranger queries {n_str}")
    print(f"{'thr':>5} | stranger wrongly matched (95% CI) | enrolled: right / wrong person / rejected")
    for t in THRESHOLDS:
        lo, hi = wilson(strangers[t], n_str)
        right, wrong, rej = (gen[t][0] / n_gen, gen[t][1] / n_gen, gen[t][2] / n_gen)
        rows_out.append({"threshold": t, "stranger_false_match": strangers[t] / n_str, "stranger_ci95": [lo, hi],
                         "enrolled_correct": right, "enrolled_wrong_person": wrong, "enrolled_rejected": rej})
        flag = "  <- app default" if abs(t - Config.FACE_MATCH_THRESHOLD) < 1e-9 else ""
        print(f"{t:5.2f} | {strangers[t] / n_str:7.2%} ({lo:.2%}-{hi:.2%}) | {right:6.1%} / {wrong:5.1%} / {rej:6.1%}{flag}")
    if a.out:
        a.out.write_text(json.dumps({"people": len(people), "photos": int(len(ids)), "splits": a.splits, "enrolled_queries": n_gen,
                                     "stranger_queries": n_str, "rows": rows_out}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

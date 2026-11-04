"""Leave-one-photo-out identification accuracy on a folder-per-person dataset.

    python scripts/evaluate_identification.py DATASET_DIR [--out results.json]

Every photo is used once as the query against a gallery made of all OTHER photos (so the query photo is
never in the gallery). Reported: top-1 / top-3 accuracy of naming the right person, and, at the app's
match threshold, how often an answer is given and how often a given answer is right."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.faces import FaceEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    eng = FaceEngine(Config.MODELS_DIR, Config.FACE_MATCH_THRESHOLD)
    vecs, ids, skipped = [], [], 0
    for folder in sorted(p for p in a.dataset.iterdir() if p.is_dir()):
        for f in sorted(folder.iterdir()):
            img = cv2.imread(str(f)) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else None
            face = eng.largest_face(img) if img is not None else None
            e = eng.embed(img, face) if face is not None else None
            if e is None:
                skipped += 1
                continue
            vecs.append(e); ids.append(folder.name)
    X, ids = np.stack(vecs), np.array(ids)
    S = X @ X.T
    np.fill_diagonal(S, -2)  # a photo never matches itself
    top1 = top3 = answered = right = 0
    n = 0
    for i in range(len(ids)):
        if (ids == ids[i]).sum() < 2:
            continue  # nobody else to match against
        n += 1
        best: dict[str, float] = {}
        for j in np.argsort(-S[i]):
            if ids[j] not in best:
                best[ids[j]] = float(S[i, j])
            if len(best) == 3:
                break
        ranked = list(best.items())
        top1 += ranked[0][0] == ids[i]
        top3 += ids[i] in [r[0] for r in ranked]
        if ranked[0][1] >= Config.FACE_MATCH_THRESHOLD:
            answered += 1
            right += ranked[0][0] == ids[i]
    out = {"people": int(len(set(ids))), "queries": n, "skipped_no_face": skipped, "top1": top1 / n, "top3": top3 / n,
           "threshold": Config.FACE_MATCH_THRESHOLD, "answered": answered / n, "correct_when_answered": right / max(answered, 1)}
    print(f"{out['people']} people, {n} queries ({skipped} photos skipped: no face)")
    print(f"top-1 {out['top1']:.1%}  top-3 {out['top3']:.1%}")
    print(f"at threshold {out['threshold']}: answers given for {out['answered']:.1%} of queries, right {out['correct_when_answered']:.1%} of the time")
    if a.out:
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

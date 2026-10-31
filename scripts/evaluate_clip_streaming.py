"""Score the *production* streaming rule on cached clip features.

    python scripts/evaluate_clip_streaming.py FEATS_DIR [--split val] [--out results.json]

The classifier is evaluated clip-by-clip in training (one score per whole clip). Live, it runs as a
rolling window and an alert needs two consecutive scores above threshold (see services/clip_model.py).
This replays each clip's measured rows through that exact ClipStream class and counts a clip as
detected if an alert would have fired at any point. The streaming parameters were fixed before this
was run; this is a second look at the validation clips and is reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.clip_model import ClipModel, ClipStream  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("feats_dir", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    model = ClipModel(Config.MODELS_DIR / Config.CLIP_MODEL, True, Config.CLIP_THRESHOLD)
    if not model.available:
        raise SystemExit(model.error)
    y, fired, final = [], [], []
    for f in sorted(a.feats_dir.glob(f"{a.split}_*.npz")):
        d = np.load(f)
        off = np.concatenate([[0], np.cumsum(d["lengths"])])
        for i in range(len(d["lengths"])):
            s, ever = ClipStream(model), False
            for row in d["frames"][off[i]:off[i + 1]]:
                s.add_row(row)
                ever = ever or s.violent
            y.append(int(d["labels"][i])); fired.append(int(ever)); final.append(s.prob)
    y, fired = np.array(y), np.array(fired)
    tp, tn = int(((fired == 1) & (y == 1)).sum()), int(((fired == 0) & (y == 0)).sum())
    fp, fn = int(((fired == 1) & (y == 0)).sum()), int(((fired == 0) & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"split": a.split, "clips": int(len(y)), "threshold": model.threshold, "accuracy": (tp + tn) / len(y),
           "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
           "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    print(f"{a.split}: {len(y)} clips | production streaming rule (threshold {model.threshold:.2f}, 2 consecutive scores)")
    print(f"  accuracy {out['accuracy']:.1%}  precision {prec:.1%}  recall {rec:.1%}  F1 {out['f1']:.2f}")
    print(f"  confusion: TP {tp}  FP {fp}  TN {tn}  FN {fn}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

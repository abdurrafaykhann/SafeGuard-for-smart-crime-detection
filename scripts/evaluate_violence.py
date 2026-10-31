"""Measure the violence detector on a labelled video dataset.

    python scripts/evaluate_violence.py PATH/TO/DATASET [--limit 100] [--out results.json]

Expected layout (RWF-2000 style; any dataset can be arranged this way):

    DATASET/
      Fight/      *.mp4 *.avi ...      <- positive class
      NonFight/   *.mp4 *.avi ...      <- negative class

Folder names are matched case-insensitively; anything containing "non" is negative, anything
else containing "fight" or "violen" is positive. Every video gets one score: the highest
smoothed risk the detector reached. The script then reports, at the alert threshold you
actually run in production AND across a threshold sweep: accuracy, precision, recall, F1,
the confusion matrix and ROC-AUC, so the README can quote real numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.violence import ViolenceDetector, YoloBackend  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SAMPLE_FPS = 5.0


def label_of(folder: str) -> int | None:
    name = folder.lower()
    if "non" in name:
        return 0
    if "fight" in name or "violen" in name:
        return 1
    return None


def score_video(path: Path, backend: YoloBackend | None, width: int) -> tuple[float, bool]:
    """Returns (max smoothed risk, whether the detector would have fired an alert)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot decode")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(fps / SAMPLE_FPS)), 1)
    det = ViolenceDetector(Config.VIOLENCE_THRESHOLD, Config.VIOLENCE_MIN_FRAMES, backend)
    best, fired, i = 0.0, False, 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)))
            res = det.analyze(frame)
            best, fired = max(best, res.smoothed), fired or res.violent
        i += 1
    cap.release()
    return best, fired


def metrics(y: list[int], pred: list[int]) -> dict:
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / max(len(y), 1),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def roc_auc(y: list[int], s: list[float]) -> float | None:
    pos = [v for t, v in zip(y, s) if t == 1]
    neg = [v for t, v in zip(y, s) if t == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)  # Mann-Whitney U
    return wins / (len(pos) * len(neg))


def evaluate(dataset: Path, limit: int | None, use_model: bool) -> dict:
    backend = YoloBackend(Config.MODELS_DIR, Config.YOLO_WEIGHTS, Config.YOLO_IMGSZ, Config.TORCH_THREADS) if use_model else None
    items: list[tuple[Path, int]] = []
    for sub in sorted(p for p in dataset.rglob("*") if p.is_dir()):
        lab = label_of(sub.name)
        if lab is None:
            continue
        vids = sorted(v for v in sub.iterdir() if v.suffix.lower() in VIDEO_EXT)
        items += [(v, lab) for v in (vids[:limit] if limit else vids)]
    if not items:
        raise SystemExit(f"No labelled videos found under {dataset} (expected Fight/ and NonFight/ folders).")

    y, scores, fired, failed = [], [], [], 0
    start = time.time()
    for n, (path, lab) in enumerate(items, 1):
        try:
            s, f = score_video(path, backend, Config.FRAME_WIDTH)
        except Exception as exc:
            failed += 1
            print(f"  skip {path.name}: {exc}")
            continue
        y.append(lab); scores.append(s); fired.append(int(f))
        if n % 10 == 0 or n == len(items):
            print(f"  {n}/{len(items)} videos ({time.time() - start:.0f}s)", flush=True)

    sweep = []
    for t in [i / 20 for i in range(1, 20)]:
        sweep.append({"threshold": t, **metrics(y, [int(s >= t) for s in scores])})
    best = max(sweep, key=lambda r: r["f1"])
    return {
        "mode": "model+heuristics" if backend and backend.available else "heuristics-only",
        "videos": len(y), "positives": sum(y), "negatives": len(y) - sum(y), "skipped": failed,
        "at_production_settings": {"threshold": Config.VIOLENCE_THRESHOLD, **metrics(y, fired)},
        "roc_auc": roc_auc(y, scores), "best_f1_threshold": best, "sweep": sweep,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--limit", type=int, help="max videos per class (quick runs)")
    ap.add_argument("--out", type=Path, help="write full results as JSON")
    ap.add_argument("--no-model", action="store_true", help="evaluate heuristics only")
    a = ap.parse_args()
    r = evaluate(a.dataset, a.limit, not a.no_model)
    p = r["at_production_settings"]
    print(f"\nmode: {r['mode']} | {r['videos']} videos ({r['positives']} fight / {r['negatives']} non-fight)")
    print(f"AT PRODUCTION SETTINGS (threshold {p['threshold']}, {Config.VIOLENCE_MIN_FRAMES} consecutive frames):")
    print(f"  accuracy {p['accuracy']:.1%}  precision {p['precision']:.1%}  recall {p['recall']:.1%}  F1 {p['f1']:.2f}")
    print(f"  confusion: TP {p['tp']}  FP {p['fp']}  TN {p['tn']}  FN {p['fn']}")
    auc = r["roc_auc"]
    print(f"ROC-AUC: {auc:.3f}" if auc is not None else "ROC-AUC: n/a (need both classes)")
    b = r["best_f1_threshold"]
    print(f"best-F1 threshold on this set: {b['threshold']:.2f} (F1 {b['f1']:.2f}) - tuned on the test data, so optimistic")
    if a.out:
        a.out.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

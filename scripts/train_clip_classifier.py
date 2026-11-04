"""Train and evaluate the clip-level violence classifier from extracted features.

    python scripts/train_clip_classifier.py FEATS_DIR --out models/violence_clip.joblib

FEATS_DIR holds train_*.npz and val_*.npz from extract_clip_features.py. Discipline that keeps the
reported number honest:
  * model family, hyper-parameters and the alert threshold are chosen ONLY from 5-fold cross-validation
    on the training clips (out-of-fold predictions);
  * the validation clips are scored exactly once, at the very end, with those frozen choices.

Optional site adaptation (see scripts/record_negatives.py):
  --extra-negatives A.npz ...   your own recordings of normal behaviour, added to training as non-violent
  --site-test B.npz             a SEPARATE recording, never trained on; reports how often the live
                                streaming rule would false-alarm on it, before vs after retraining
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from safeguard.services.clip_features import CLIP_FEATURES, FRAME_FEATURES, clip_vector  # noqa: E402
from safeguard.services.clip_model import ClipStream  # noqa: E402


def load_split(feats_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, names = [], [], []
    for f in sorted(feats_dir.glob(f"{split}_*.npz")):
        d = np.load(f)
        off = np.concatenate([[0], np.cumsum(d["lengths"])])
        for i in range(len(d["lengths"])):
            X.append(clip_vector(d["frames"][off[i]:off[i + 1]]))
            y.append(int(d["labels"][i])); names.append(str(d["names"][i]))
    return np.stack(X), np.array(y), names


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d["lengths"])])
    X = [clip_vector(d["frames"][off[i]:off[i + 1]]) for i in range(len(d["lengths"]))]
    return np.stack(X), d["labels"].astype(int)


def site_false_alarm_rate(path: Path, model, threshold: float) -> tuple[int, int]:
    """(clips that would have raised a live alert, clips) using the exact streaming rule from services/clip_model."""
    class _M:
        available = True

        def __init__(self):
            self.threshold = threshold

        def prob(self, rows):
            return float(model.predict_proba(clip_vector(rows)[None])[0, 1])

    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d["lengths"])])
    fired = 0
    for i in range(len(d["lengths"])):
        s, ever = ClipStream(_M()), False
        for row in d["frames"][off[i]:off[i + 1]]:
            s.add_row(row)
            ever = ever or s.violent
        fired += int(ever)
    return fired, len(d["lengths"])


def candidates() -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "extra_trees": ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "hist_gb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=0),
    }


def metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = (p >= t).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": round(float(t), 3), "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Maximise F1 on out-of-fold training predictions (balances missed fights against false alarms)."""
    grid = np.linspace(0.1, 0.9, 81)
    return float(max(grid, key=lambda t: metrics(y, p, t)["f1"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("feats_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models/violence_clip.joblib"))
    ap.add_argument("--report", type=Path, default=Path("docs/training_violence_clip.json"))
    ap.add_argument("--no-val", action="store_true", help="skip the final validation scoring")
    ap.add_argument("--extra-negatives", type=Path, nargs="*", default=[], help="npz files of your own normal behaviour")
    ap.add_argument("--site-test", type=Path, help="held-out npz of normal behaviour, never trained on")
    ap.add_argument("--compare-with", type=Path, help="earlier model to compare against on --site-test (default: --out)")
    ap.add_argument("--extra-repeat", type=int, default=3, help="repeat each of your clips this many times so a few dozen "
                    "clips are not drowned out by 700 dataset clips")
    a = ap.parse_args()

    Xtr, ytr, _ = load_split(a.feats_dir, "train")
    old_bundle = None
    compare = a.compare_with or a.out
    if a.site_test and compare.exists():
        old_bundle = joblib.load(compare)  # our own earlier artefact, kept for the before/after comparison
    for extra in a.extra_negatives:
        Xe, ye = load_npz(extra)
        assert (ye == 0).all(), "extra files must contain only non-violent clips"
        Xtr, ytr = np.vstack([Xtr] + [Xe] * a.extra_repeat), np.concatenate([ytr] + [ye] * a.extra_repeat)
        print(f"added {len(ye)} of your own non-violent clips from {extra.name}")
    print(f"train clips: {len(ytr)} ({int(ytr.sum())} fight / {int((1 - ytr).sum())} non-fight) | {Xtr.shape[1]} features")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    results = {}
    for name, model in candidates().items():
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        results[name] = {"auc": float(roc_auc_score(ytr, oof)), "oof": oof, **metrics(ytr, oof, 0.5)}
        print(f"  CV {name:14s} AUC {results[name]['auc']:.3f} | acc@0.5 {results[name]['accuracy']:.3f}")
    best = max(results, key=lambda k: results[k]["auc"])
    thr = pick_threshold(ytr, results[best]["oof"])
    cv_at_thr = metrics(ytr, results[best]["oof"], thr)
    print(f"selected by CV AUC: {best} | threshold {thr:.2f} (max out-of-fold F1) | CV acc {cv_at_thr['accuracy']:.3f}")

    model = candidates()[best].fit(Xtr, ytr)
    report = {"model": best, "sklearn": sklearn.__version__, "train_clips": int(len(ytr)),
              "cv": {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in results.items()},
              "threshold": thr, "cv_at_threshold": cv_at_thr}

    if not a.no_val:
        Xva, yva, _ = load_split(a.feats_dir, "val")
        pva = model.predict_proba(Xva)[:, 1]
        val = metrics(yva, pva, thr)
        val["auc"] = float(roc_auc_score(yva, pva)); val["clips"] = int(len(yva))
        report["validation_once"] = val
        print(f"\nVALIDATION ({len(yva)} unseen clips, scored once): acc {val['accuracy']:.3f} | precision {val['precision']:.3f} | "
              f"recall {val['recall']:.3f} | F1 {val['f1']:.3f} | AUC {val['auc']:.3f}")
        print(f"  confusion: TP {val['tp']}  FP {val['fp']}  TN {val['tn']}  FN {val['fn']}")

    if a.site_test:
        new_fa = site_false_alarm_rate(a.site_test, model, thr)
        report["site_test"] = {"file": a.site_test.name, "new_false_alarms": new_fa[0], "clips": new_fa[1]}
        print(f"\nSITE TEST ({a.site_test.name}, held out, all normal behaviour): "
              f"new model would false-alarm on {new_fa[0]}/{new_fa[1]} clips", end="")
        if old_bundle is not None:
            old_fa = site_false_alarm_rate(a.site_test, old_bundle["model"], float(old_bundle["threshold"]))
            report["site_test"]["old_false_alarms"] = old_fa[0]
            print(f" | previous model: {old_fa[0]}/{old_fa[1]}")
        else:
            print()

    a.out.parent.mkdir(parents=True, exist_ok=True); a.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "features": CLIP_FEATURES, "frame_features": FRAME_FEATURES,
                 "fps": 4.0, "max_frames": 20, "sklearn": sklearn.__version__}, a.out)
    a.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", a.out, "and", a.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

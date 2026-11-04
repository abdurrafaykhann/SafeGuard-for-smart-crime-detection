"""Score a weapon detector on YOUR held-out webcam frames.

    python scripts/weapons/eval_webcam.py models/Weapon_yolo11n.onnx models/Weapon_threat_yolov8n.pt ...

Sets (all under instance/weapons_site/, git-ignored, never trained on):
  knife_test  frames of you holding a knife; knife_test_labels.json lists which frame indices clearly show
              the knife (K), clearly show none (N); ambiguous frames are excluded from scoring.
              The labels were assigned by visual inspection, so treat small differences as noise.
  negatives_test  100 frames of normal activity with look-alike objects, all counted as no-weapon frames.

For each model and confidence level it reports
  knife recall      share of K frames where a `knife` box is found,
  any-weapon recall share of K frames where ANY weapon class fires (looser),
  false-alarm rate  share of no-weapon frames (N + negatives_test) where ANY weapon class fires.
A usable detector needs high recall with a false-alarm rate near zero; both numbers matter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2  # noqa: E402

from safeguard.config import Config  # noqa: E402  (sets OMP policy before torch loads)

ROOT = Path(__file__).resolve().parent.parent.parent
SITE = ROOT / "instance" / "weapons_site"
WEAPON_WORDS = ("knife", "gun", "pistol", "rifle", "handgun", "shotgun", "firearm", "grenade", "weapon")
THRESHOLDS = (0.25, 0.40, 0.50, 0.60)


def frames():
    files = sorted((SITE / "knife_test").glob("*.jpg"))
    labels = json.loads((SITE / "knife_test_labels.json").read_text())
    pos = [files[i] for i in labels["knife"]]
    neg = [files[i] for i in labels["no_knife"]] + sorted((SITE / "negatives_test").glob("*.jpg"))
    return pos, neg


def detect_all(model, paths, imgsz):
    out = []
    for p in paths:
        r = model.predict(cv2.imread(str(p)), conf=0.05, imgsz=imgsz, verbose=False)[0]
        out.append([(str(model.names[int(c)]).lower(), float(s)) for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())])
    return out


def main() -> int:
    from ultralytics import YOLO
    pos, neg = frames()
    print(f"knife frames (K): {len(pos)} | no-weapon frames: {len(neg)}\n")
    results = {}
    for path in sys.argv[1:]:
        model = YOLO(path)
        imgsz = 640
        det_pos, det_neg = detect_all(model, pos, imgsz), detect_all(model, neg, imgsz)
        print(f"== {Path(path).name}  classes: {list(model.names.values())}")
        print(f"   {'conf':>5s} | knife recall | any-weapon recall | false-alarm rate on no-weapon frames")
        rows = []
        for t in THRESHOLDS:
            kr = sum(any(n == "knife" and c >= t for n, c in d) for d in det_pos) / len(pos)
            ar = sum(any(any(w in n for w in WEAPON_WORDS) and c >= t for n, c in d) for d in det_pos) / len(pos)
            fa = sum(any(any(w in n for w in WEAPON_WORDS) and c >= t for n, c in d) for d in det_neg) / len(neg)
            rows.append({"conf": t, "knife_recall": round(kr, 3), "any_weapon_recall": round(ar, 3), "false_alarm_rate": round(fa, 3)})
            print(f"   {t:5.2f} | {kr:12.0%} | {ar:17.0%} | {fa:.0%}")
        results[Path(path).name] = rows
        print()
    out = ROOT / "docs" / "weapon_webcam_eval.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

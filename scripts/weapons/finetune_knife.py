"""Fine-tune a community weapon detector on the knife-focused subset, on CPU, within a time budget.

    python scripts/weapons/finetune_knife.py [BASE.pt] [--hours 1.5] [--imgsz 416]

Base default: models/Weapon_threat_yolov8n.pt (Subh775/Threat-Detection-YOLOv8n, MIT). Data: instance/weapons_ft (make_knife_subset.py).
Strong scale augmentation imitates close-ups. Results land in instance/weapons_ft_runs/knife_ft; the best weights are also exported
to ONNX. The held-out webcam frames are NOT used here: score the result with scripts/weapons/eval_webcam.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from safeguard.config import Config  # noqa: E402,F401  (sets OMP policy before torch loads)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default=str(ROOT / "models" / "Weapon_threat_yolov8n.pt"))
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(a.threads)
    model = YOLO(a.base)
    model.train(data=str(ROOT / "instance" / "weapons_ft" / "data.yaml"), imgsz=a.imgsz, epochs=40, time=a.hours, batch=16, workers=2, device="cpu",
                project=str(ROOT / "instance" / "weapons_ft_runs"), name="knife_ft", exist_ok=True, scale=0.9, mosaic=1.0, close_mosaic=3, fliplr=0.5,
                hsv_v=0.5, degrees=5, patience=50, plots=False, amp=False, lr0=0.005, verbose=True)
    best = ROOT / "instance" / "weapons_ft_runs" / "knife_ft" / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, simplify=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

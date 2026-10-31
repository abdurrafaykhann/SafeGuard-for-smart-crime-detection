"""Load the optional weapon model and sanity-check it on your webcam.

    python scripts/check_weapon_model.py

Prints the model's classes, which of them SafeGuard treats as weapons, and how many
detections it makes over ~5 seconds of live webcam frames. In a normal scene that should
be zero; anything else is a false positive worth knowing about before you rely on it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from safeguard.config import Config  # noqa: E402
from safeguard.services.violence import ViolenceDetector, WeaponBackend, YoloBackend  # noqa: E402


def main() -> int:
    wb = WeaponBackend(Config.MODELS_DIR / Config.WEAPON_WEIGHTS)
    print("available:", wb.available, "| error:", wb.error)
    if not wb.available:
        return 1
    print("model classes:", wb.model.names)
    print("counted as weapons:", wb.ids)
    det = ViolenceDetector(backend=YoloBackend(Config.MODELS_DIR, Config.YOLO_WEIGHTS), weapons=wb)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    if ok:  # the first inference pays a one-off warm-up cost; keep it out of the timing
        det.analyze(frame)
    frames, hits, start = 0, [], time.time()
    while time.time() - start < 5:
        ok, frame = cap.read()
        if not ok:
            break
        hits += det.analyze(frame).weapons
        frames += 1
    cap.release()
    print(f"webcam: {frames} frames in 5s ({frames / 5:.1f} fps with both models)")
    print(f"weapon detections: {len(hits)}", hits[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())

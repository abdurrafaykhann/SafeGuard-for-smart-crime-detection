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


COUNTDOWN_S = 10  # time to get an object into frame
CAPTURE_S = 15


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

    print(f"Get your object ready: capture starts in {COUNTDOWN_S}s and runs for {CAPTURE_S}s.", flush=True)
    end = time.time() + COUNTDOWN_S
    while time.time() < end:  # keep reading so the camera buffer stays fresh
        cap.read()
    print("CAPTURE NOW", flush=True)

    frames, hits, alerts, start = 0, [], 0, time.time()
    best: dict[str, float] = {}
    while time.time() - start < CAPTURE_S:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.analyze(frame)
        frames += 1
        alerts += res.weapon_alert
        for name, conf, _ in res.weapons:
            hits.append((name, conf))
            best[name] = max(best.get(name, 0.0), conf)
    cap.release()
    print(f"webcam: {frames} frames in {CAPTURE_S}s ({frames / CAPTURE_S:.1f} fps with both models)")
    print(f"weapon detections: {len(hits)} across {frames} frames")
    print("best confidence per class:", {k: round(v, 2) for k, v in best.items()} or "none")
    print(f"frames where SafeGuard would raise a weapon alert (sustained, conf>=0.55): {alerts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

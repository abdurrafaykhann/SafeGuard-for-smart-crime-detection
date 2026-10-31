"""Capture YOUR webcam frames for training / testing the weapon detector.

    python scripts/weapons/capture_webcam.py negatives --frames 300
    python scripts/weapons/capture_webcam.py knife-test --frames 150

Frames are saved as JPEGs under instance/weapons_site/<mode>/ (git-ignored, local only; delete the
folder any time). A beep marks each change of prompt, and three beeps mark the start.

  negatives   Your normal scene with NO weapon in view: teaches the model your background, desk and hands
              are not weapons. Includes harmless look-alikes (pen, phone, remote, spoon, scissors held
              closed ...). Used for TRAINING as images with no boxes.
  knife-test  You holding a kitchen knife SAFELY (blade pointed away from the camera and from you, held
              firmly, no sudden movements) at varied distances and angles. NEVER trained on: it is the
              held-out real-world check. No box labels are needed because it is scored per frame
              ("did the detector find a knife in this frame?").
  negatives-test  A second, separate negatives session, also held out, to measure false alarms.

Never wave a real blade near people or the camera. If in doubt, skip knife-test.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS = {
    "negatives": ["sit and type", "hold a pen", "hold your phone", "hold a TV remote", "hold a spoon or spatula",
                  "hold a mug", "hold a book", "hold scissors CLOSED, blades down", "wave both hands",
                  "lean in close to the camera", "stand up and walk away and back", "adjust the lighting / turn your lamp"],
    "negatives-test": ["sit and type", "hold a pen", "hold your phone", "hold a mug", "hold a book", "wave both hands",
                       "lean in close", "stand and stretch", "hold a spoon", "hold a remote"],
    "knife-test": ["hold the knife at chest height, blade away", "hold it near the camera, flat", "hold it side-on",
                   "hold it further back, arm extended", "tilt it slowly", "hold it with both hands, low",
                   "partly hidden behind your hand", "hold it up beside your face (blade away from you)"],
}


def beep(n: int = 1) -> None:
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(880, 140)
            time.sleep(0.08)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(PROMPTS))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--countdown", type=int, default=15)
    ap.add_argument("--per-second", type=float, default=2.0)
    ap.add_argument("--prompt-seconds", type=float, default=8.0)
    a = ap.parse_args()

    out = ROOT / "instance" / "weapons_site" / a.mode.replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("cannot open camera", a.camera)
        return 1
    for _ in range(15):
        cap.read()
    print(f"{a.mode}: {a.frames} frames, starting in {a.countdown}s. Prompts change every {a.prompt_seconds:.0f}s:")
    for i, p in enumerate(PROMPTS[a.mode], 1):
        print(f"  {i}. {p}")
    end = time.time() + a.countdown
    while time.time() < end:
        cap.read()
    beep(3)

    prompts, saved, last_prompt, t_last = PROMPTS[a.mode], 0, -1, 0.0
    start = time.time()
    while saved < a.frames:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time() - start
        k = int(now // a.prompt_seconds)
        if k != last_prompt:
            last_prompt = k
            print(f"-> {prompts[k % len(prompts)]}", flush=True)
            if k:
                beep(1)
        if now - t_last >= 1.0 / a.per_second:
            t_last = now
            cv2.imwrite(str(out / f"{a.mode}_{int(time.time())}_{saved:04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
    cap.release()
    print(f"saved {saved} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Record YOUR cameras' normal behaviour as extra "non-violent" training examples.

    python scripts/record_negatives.py --minutes 4 --out feats_site_a.npz

The detector was trained on street/crowd footage, so ordinary movement in front of a close-up webcam can
score as violence. This captures that movement from the live camera and keeps only the numeric
per-frame measurements (motion, optical flow, YOLO confidence...). No images or video are written
to disk. Everything captured is labelled non-violent, so ONLY perform normal activity while it
runs: typing, stretching, waving, leaning in, standing up, walking past, picking things up, gesturing.
(The script prints a prompt for each 5-second clip so you can vary what you do.)

Then retrain with the data added:
    python scripts/train_clip_classifier.py FEATS_DIR --extra-negatives feats_site_a.npz ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from safeguard.config import Config  # noqa: E402  (sets OMP_WAIT_POLICY before torch loads)
import cv2  # noqa: E402

from safeguard.services.clip_features import FrameFeaturizer  # noqa: E402
from safeguard.services.violence import YoloBackend  # noqa: E402

PROMPTS = ["sit and type naturally", "stretch your arms up", "wave at the camera", "lean in close, then back",
           "stand up and sit down", "walk across the room", "pick something up and put it down", "talk and gesture",
           "turn around", "rub your face / adjust your hair", "clap or fidget with your hands", "move your chair"]


def _beep(n: int = 1) -> None:
    """Audible cue so you do not have to watch the screen (Windows only; silent elsewhere)."""
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(880, 140)
            time.sleep(0.08)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--countdown", type=int, default=8)
    a = ap.parse_args()

    yolo = YoloBackend(Config.MODELS_DIR, Config.YOLO_WEIGHTS, Config.YOLO_IMGSZ, Config.TORCH_THREADS)
    if not yolo.available:
        print("YOLO model unavailable:", yolo.error, "- features would not match training; aborting.")
        return 1
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        print("cannot open camera", a.camera)
        return 1
    for _ in range(15):
        cap.read()

    n_clips = max(1, int(a.minutes * 60 / a.clip_seconds))
    print(f"Recording {n_clips} clips x {a.clip_seconds:.0f}s of NORMAL activity. Starting in {a.countdown}s...", flush=True)
    end = time.time() + a.countdown
    while time.time() < end:
        cap.read()

    seqs, sample_dt = [], 0.25
    _beep(3)  # three beeps = recording starts now
    for k in range(n_clips):
        print(f"clip {k + 1}/{n_clips}: {PROMPTS[k % len(PROMPTS)]}", flush=True)
        if k:
            _beep(1)  # one beep = next prompt (the 12 prompts repeat in the order listed in PROMPTS)
        feat, rows, last, t0 = FrameFeaturizer(yolo), [], -1.0, time.time()
        while time.time() - t0 < a.clip_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time() - t0
            if now - last >= sample_dt - 0.02:
                h, w = frame.shape[:2]
                if w > Config.FRAME_WIDTH:
                    frame = cv2.resize(frame, (Config.FRAME_WIDTH, int(h * Config.FRAME_WIDTH / w)))
                rows.append(feat(frame))
                last = now
        seqs.append(np.stack(rows))
    cap.release()

    lengths = np.array([len(s) for s in seqs])
    np.savez_compressed(a.out, frames=np.concatenate(seqs), lengths=lengths, labels=np.zeros(len(seqs), int),
                        names=np.array([f"site_{k}" for k in range(len(seqs))]))
    print(f"saved {len(seqs)} clips ({int(lengths.sum())} frames of numbers only, no images) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

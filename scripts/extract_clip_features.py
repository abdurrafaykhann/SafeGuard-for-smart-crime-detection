"""Turn labelled videos into per-frame feature sequences for training the clip classifier.

    python scripts/extract_clip_features.py DATASET/train --out feats_train_0.npz --shard 0 --of 4 --limit 20

Each shard handles every Nth video so several processes can share the CPU. Output is one .npz with
the concatenated frame features, the per-clip lengths, labels and file names."""
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

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(root: Path) -> list[tuple[Path, int]]:
    items = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        name = sub.name.lower()
        label = 0 if "non" in name else 1 if ("fight" in name or "violen" in name) else None
        if label is None:
            continue
        items += [(v, label) for v in sorted(sub.iterdir()) if v.suffix.lower() in VIDEO_EXT]
    return items


def featurize(path: Path, yolo: YoloBackend | None, fps: float, max_frames: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot decode")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(src_fps / fps)), 1)
    feat, rows, i = FrameFeaturizer(yolo), [], 0
    while cap.grab() and len(rows) < max_frames:
        if i % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)))
            rows.append(feat(frame))
        i += 1
    cap.release()
    return np.stack(rows) if rows else np.zeros((0, 12), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, help="max videos per class (quick runs)")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()

    items = list_videos(a.dataset)
    if a.limit:
        per = {0: 0, 1: 0}
        kept = []
        for p, lab in items:
            if per[lab] < a.limit:
                kept.append((p, lab)); per[lab] += 1
        items = kept
    items = items[a.shard::a.of]
    yolo = YoloBackend(Config.MODELS_DIR, Config.YOLO_WEIGHTS, Config.YOLO_IMGSZ, a.threads)
    print(f"shard {a.shard}/{a.of}: {len(items)} clips | yolo: {'on' if yolo.available else 'OFF (' + str(yolo.error) + ')'}", flush=True)
    cv2.setNumThreads(a.threads)

    seqs, labels, names, start = [], [], [], time.time()
    for n, (path, lab) in enumerate(items, 1):
        try:
            seqs.append(featurize(path, yolo if yolo.available else None, a.fps, a.max_frames, Config.FRAME_WIDTH))
            labels.append(lab); names.append(path.name)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}", flush=True)
        if n % 10 == 0 or n == len(items):
            print(f"  shard {a.shard}: {n}/{len(items)} ({time.time() - start:.0f}s)", flush=True)
    lengths = np.array([len(s) for s in seqs])
    np.savez_compressed(a.out, frames=np.concatenate(seqs) if seqs else np.zeros((0, 12), np.float32),
                        lengths=lengths, labels=np.array(labels), names=np.array(names))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

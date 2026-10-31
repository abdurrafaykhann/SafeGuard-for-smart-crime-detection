"""Per-frame measurements and clip-level summaries for the trained violence classifier.

The same code builds the training set and runs live, so the classifier never sees a feature
computed differently from how it was trained. Frame features combine what the YOLO violence model
sees (violence confidence, number of people, contact) with motion statistics from dense optical flow
and a simple injury-colour fraction."""
from __future__ import annotations

import cv2
import numpy as np

FRAME_FEATURES = [
    "yolo_violence", "yolo_persons", "contact",              # from the fine-tuned YOLO model
    "moving_fraction", "abs_diff",                            # frame differencing
    "flow_mean", "flow_p90", "flow_std", "flow_coherence",    # Farneback optical flow
    "motion_spread", "red_fraction", "brightness",
]
N_FRAME = len(FRAME_FEATURES)
FLOW_SIZE = (160, 120)


class FrameFeaturizer:
    """Stateful: keeps the previous frame to measure motion. One instance per clip / camera."""

    def __init__(self, yolo=None) -> None:
        self.yolo = yolo if (yolo is not None and getattr(yolo, "available", False)) else None
        self._prev_small: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        from .violence import ViolenceDetector  # local import: avoids a cycle at module load

        out = np.zeros(N_FRAME, dtype=np.float32)
        if self.yolo is not None:
            conf, persons = self.yolo.infer(frame)
            out[0], out[1], out[2] = conf, len(persons), ViolenceDetector._contact_cue(persons)

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        small = cv2.resize(gray, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            out[3] = np.count_nonzero(diff > 25) / diff.size
            out[4] = diff.mean() / 255.0
            flow = cv2.calcOpticalFlowFarneback(self._prev_small, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            out[5], out[6], out[7] = mag.mean(), np.percentile(mag, 90), mag.std()
            # coherence: 1 when everything moves the same way (a walking crowd), ~0 when motion is chaotic
            total = mag.sum()
            out[8] = np.hypot(flow[..., 0].sum(), flow[..., 1].sum()) / total if total > 1e-6 else 0.0
            if total > 1e-6:  # how spread out across the picture the motion is
                ys, xs = np.mgrid[0:mag.shape[0], 0:mag.shape[1]]
                w = mag / total
                cx, cy = (w * xs).sum(), (w * ys).sum()
                out[9] = np.sqrt((w * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum()) / FLOW_SIZE[0]
        self._prev_gray, self._prev_small = gray, small

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        out[10] = np.count_nonzero(((h < 8) | (h > 172)) & (s > 150) & (v > 60) & (v < 230)) / h.size
        out[11] = v.mean() / 255.0
        return out


def clip_vector(rows: np.ndarray) -> np.ndarray:
    """Summarise a (T, N_FRAME) sequence into one fixed-length vector: level, spread, peaks, trend."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or len(rows) < 2:
        return np.zeros(N_FRAME * 6 + 1, dtype=np.float32)
    rows = rows[1:]  # first frame has no motion reference
    k = min(3, len(rows))
    t = np.linspace(-1, 1, len(rows), dtype=np.float32)
    parts = [
        rows.mean(0), rows.std(0), rows.max(0), np.percentile(rows, 90, axis=0),
        np.sort(rows, axis=0)[-k:].mean(0),                       # mean of the k highest frames
        ((rows - rows.mean(0)) * t[:, None]).mean(0),             # linear trend across the clip
        [float(len(rows))],
    ]
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


CLIP_FEATURES = ([f"{n}_{stat}" for stat in ("mean", "std", "max", "p90", "top3", "trend") for n in FRAME_FEATURES]
                 + ["n_frames"])

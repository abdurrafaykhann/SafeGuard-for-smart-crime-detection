"""Reject corrupted camera frames before any model sees them.

A glitching USB webcam or a dropped network stream can hand over frames that are mostly digital noise. On one
demo camera such frames produced weapon alerts (e.g. "rifle 59%") because the detectors will happily "find" shapes
in noise. Measured on the 104 snapshots saved by that demo: normal alert frames had a Laplacian variance of at most
331, the corrupted ones 9,947 to 67,520. The limit sits in the gap, with a wide margin on both sides."""
from __future__ import annotations

import cv2
import numpy as np

NOISE_LIMIT = 3000.0   # Laplacian variance of the grey frame (at <= 640 px wide) above which it is treated as corrupted
_REF_WIDTH = 640       # the statistic depends on resolution, so always measure at the width it was calibrated on


def noise_level(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    if w > _REF_WIDTH:
        frame = cv2.resize(frame, (_REF_WIDTH, int(h * _REF_WIDTH / w)), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def is_corrupt(frame: np.ndarray) -> bool:
    return noise_level(frame) > NOISE_LIMIT

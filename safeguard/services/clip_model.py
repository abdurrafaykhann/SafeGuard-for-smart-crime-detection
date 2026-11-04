"""The trained clip-level violence classifier, run as a rolling window over a live stream.

Training (scripts/train_clip_classifier.py) looked at whole 5-second clips sampled at ~4 frames/s.
Live, the same measurements are taken every 0.25 s into a 20-row window, the classifier re-scores
the window once a second, and an alert needs TWO consecutive scores above the trained threshold.
Those streaming parameters were fixed a priori (not tuned on any evaluation data)."""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from .clip_features import FRAME_FEATURES, FrameFeaturizer, clip_vector

_MOVING = FRAME_FEATURES.index("moving_fraction")

log = logging.getLogger(__name__)

SAMPLE_S = 0.25      # one measurement every 0.25 s  (training: ~4 fps)
WINDOW = 20          # rows kept: 5 s, as in training
MIN_ROWS = 9         # first row has no motion reference, so 8 usable rows (~2 s) before scoring
EVAL_EVERY = 4       # re-score every 4 new rows (~1 s)
STREAK = 2           # consecutive scores >= threshold needed to call it violence

# Scene-wide-motion gate. When the picture as a whole changes at once (a shaken or moved camera, or
# someone filling the lens) the classifier is outside what it was trained on (RWF clips almost never
# change over >30% of the frame: 1.6% of calm rows, 4.2% of fight rows). Alerts are held while at least
# half of the last 5 s of samples are like that. Chosen from the *training* clips only.
SCENE_MOTION = 0.30  # moving_fraction at/above which one sample counts as "whole picture moving"
SCENE_SHARE = 0.5    # share of the window that must be like that before alerts are held


class ClipModel:
    """Loads the joblib bundle written by the training script. The file is our own artefact in
    models/; joblib uses pickle, so never point this at a file you did not train yourself."""

    def __init__(self, path: Path | None, enabled: bool = True, threshold_override: float = 0.0) -> None:
        self.model = None
        self.threshold = 0.5
        self.error: str | None = None
        if not enabled:
            self.error = "disabled (SAFEGUARD_CLIP_ENABLED=false)"
            return
        if path is None or not Path(path).exists():
            self.error = "no trained clip model found (run scripts/train_clip_classifier.py)"
            return
        try:
            import joblib
            bundle = joblib.load(path)
            self.model, self.threshold = bundle["model"], float(bundle["threshold"])
            self.trained_threshold = self.threshold
            if threshold_override > 0:  # per-site calibration; the trained value stays recorded for reference
                self.threshold = float(threshold_override)
            try:  # one-row predictions gain nothing from a thread pool, and spawning one every second per camera wastes CPU
                self.model.set_params(n_jobs=1)
            except Exception:
                pass
            from .clip_features import CLIP_FEATURES
            if list(bundle["features"]) != CLIP_FEATURES:
                raise ValueError("feature layout changed since this model was trained; retrain it")
        except Exception as exc:
            self.model, self.error = None, f"could not load clip model: {exc}"
            log.warning(self.error)

    @property
    def available(self) -> bool:
        return self.model is not None

    def prob(self, rows: np.ndarray) -> float:
        return float(self.model.predict_proba(clip_vector(rows)[None])[0, 1])


class ClipStream:
    """Rolling-window scorer. One instance per camera / per video."""

    def __init__(self, model: ClipModel, gate: bool = True) -> None:
        self.model = model
        self.gate = gate
        self.held = False  # True while alerts are suppressed by the scene-wide-motion gate
        self.rows: deque[np.ndarray] = deque(maxlen=WINDOW)
        self._featurizer = FrameFeaturizer(None)  # YOLO output is supplied by the caller (no second inference)
        self._last_t: float | None = None
        self._since_eval = 0
        self.prob = 0.0
        self.streak = 0

    @property
    def active(self) -> bool:
        """False until enough rows exist, so callers can fall back to the frame-level detector."""
        return self.model.available and len(self.rows) >= MIN_ROWS

    @property
    def violent(self) -> bool:
        return self.active and not self.held and self.streak >= STREAK

    def push(self, frame: np.ndarray, t: float, yolo_out: tuple[float, list]) -> None:
        if self._last_t is not None and t - self._last_t < SAMPLE_S - 0.02:
            return
        self._last_t = t
        self.add_row(self._featurizer(frame, yolo_out))

    def add_row(self, row: np.ndarray) -> None:
        """Entry point shared with offline evaluation, which replays already-measured rows."""
        self.rows.append(row)
        self._since_eval += 1
        if len(self.rows) >= MIN_ROWS and self._since_eval >= EVAL_EVERY:
            self._since_eval = 0
            self.prob = self.model.prob(np.stack(self.rows))
            self.streak = self.streak + 1 if self.prob >= self.model.threshold else 0
        if self.gate and len(self.rows) >= MIN_ROWS:
            mf = np.array([r[_MOVING] for r in self.rows])
            self.held = bool((mf >= SCENE_MOTION).mean() >= SCENE_SHARE)
            if self.held:
                self.streak = 0
        else:
            self.held = False

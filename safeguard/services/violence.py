"""Violence analysis.

Rewritten from the original ensemble. Differences that matter:
  * Every cue is a normalised 0..1 value and the final score is an explainable noisy-OR of
    weighted cues, so the UI can show *why* a frame was flagged.
  * The old code reported a hard-coded 99.5% "non-violence confidence" whenever nothing
    fired. That number was not evidence of anything, so it is gone.
  * The Canny aspect-ratio "weapon" heuristic fired on any long edge (door frames, monitors)
    and has been removed rather than disguised.
  * A trigger needs several consecutive frames above threshold, which removes single-frame
    flicker. The detector owns its own state, so one instance per camera / per video."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# weight of each cue inside the noisy-OR
WEIGHTS = {"model": 0.85, "interaction": 0.60, "blood": 0.45, "motion": 0.22}


@dataclass
class ViolenceResult:
    score: float = 0.0
    smoothed: float = 0.0
    violent: bool = False
    label: str = "Clear"
    persons: list[tuple[int, int, int, int]] = field(default_factory=list)
    cues: dict[str, float] = field(default_factory=dict)


class YoloBackend:
    """Optional. Loads lazily so the app still boots (heuristic mode) without torch."""

    def __init__(self, weights_dir: Path, weights_name: str) -> None:
        self.model = None
        self.violence_ids: set[int] = set()
        self.person_ids: set[int] = set()
        self.error: str | None = None
        try:
            from ultralytics import YOLO  # noqa: WPS433 (deliberately lazy)
        except Exception as exc:  # ImportError, or torch DLL problems on Windows
            self.error = f"ultralytics unavailable ({exc.__class__.__name__})"
            return
        for candidate in (weights_dir.parent / weights_name, weights_dir / weights_name, Path(weights_name)):
            if candidate.exists():
                try:
                    self.model = YOLO(str(candidate))
                    break
                except Exception as exc:  # corrupt / incompatible weights
                    self.error = f"could not load {candidate.name}: {exc}"
        if self.model is None:
            self.error = self.error or f"weights '{weights_name}' not found"
            return
        for idx, name in self.model.names.items():
            lname = str(name).lower()
            if "violen" in lname and "non" not in lname:
                self.violence_ids.add(int(idx))
            elif "non" in lname or lname == "person":
                self.person_ids.add(int(idx))
        if not self.violence_ids:  # original training convention: 0 = non_violence, 1 = violence
            self.violence_ids, self.person_ids = {1}, {0}

    @property
    def available(self) -> bool:
        return self.model is not None

    def infer(self, frame: np.ndarray) -> tuple[float, list[tuple[int, int, int, int]]]:
        res = self.model(frame, verbose=False, imgsz=416)[0]
        persons, best = [], 0.0
        for cls, conf, box in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
            if int(cls) in self.violence_ids:
                best = max(best, float(conf))
            if int(cls) in self.person_ids:
                persons.append(tuple(int(v) for v in box))
        return best, persons


class ViolenceDetector:
    def __init__(self, threshold: float = 0.62, min_frames: int = 3, backend: YoloBackend | None = None) -> None:
        self.threshold = threshold
        self.min_frames = min_frames
        self.backend = backend
        self._prev_gray: np.ndarray | None = None
        self._ema = 0.0
        self._streak = 0
        self._motion_hist: deque[float] = deque(maxlen=12)

    @property
    def mode(self) -> str:
        return "model+heuristics" if self.backend and self.backend.available else "heuristics-only"

    # ---- individual cues -------------------------------------------------
    def _motion_cue(self, frame: np.ndarray) -> float:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        moving_fraction = float(np.count_nonzero(diff > 25)) / diff.size
        self._motion_hist.append(moving_fraction)
        # sustained AND jerky movement scores higher than a single person walking past
        jerk = float(np.std(self._motion_hist)) if len(self._motion_hist) > 4 else 0.0
        return float(np.clip((moving_fraction - 0.04) / 0.22 + jerk * 2.0, 0.0, 1.0))

    @staticmethod
    def _blood_cue(frame: np.ndarray) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red = ((h < 8) | (h > 172)) & (s > 150) & (v > 60) & (v < 230)
        fraction = float(np.count_nonzero(red)) / red.size
        return float(np.clip((fraction - 0.03) / 0.15, 0.0, 1.0))

    @staticmethod
    def _contact_cue(persons: list[tuple[int, int, int, int]]) -> float:
        best = 0.0
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                ax1, ay1, ax2, ay2 = persons[i]
                bx1, by1, bx2, by2 = persons[j]
                iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
                if iw <= 0 or ih <= 0:
                    continue
                smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)) or 1
                best = max(best, min(1.0, (iw * ih) / smaller * 1.5))
        return best

    # ---- main entry ------------------------------------------------------
    def analyze(self, frame: np.ndarray) -> ViolenceResult:
        persons: list[tuple[int, int, int, int]] = []
        model_cue = 0.0
        if self.backend and self.backend.available:
            try:
                model_cue, persons = self.backend.infer(frame)
            except Exception as exc:
                log.warning("YOLO inference failed: %s", exc)
        motion = self._motion_cue(frame)
        blood = self._blood_cue(frame)
        contact = self._contact_cue(persons)
        interaction = float(np.clip(contact * motion * 1.6, 0.0, 1.0)) if len(persons) >= 2 else 0.0

        cues = {"model": model_cue, "interaction": interaction, "blood": blood, "motion": motion}
        # blood-coloured pixels alone are weak evidence (red walls, clothing): require motion too
        cues["blood"] = blood * (0.4 + 0.6 * motion)
        miss = 1.0
        for name, value in cues.items():
            miss *= 1.0 - WEIGHTS[name] * value
        score = 1.0 - miss

        self._ema = 0.45 * score + 0.55 * self._ema
        self._streak = self._streak + 1 if self._ema >= self.threshold else 0
        violent = self._streak >= self.min_frames
        label = "Clear"
        if violent or self._ema > self.threshold * 0.7:
            dominant = max(cues, key=lambda k: WEIGHTS[k] * cues[k])
            label = {"model": "Violence detected by model", "interaction": "Physical altercation",
                     "blood": "Possible injury", "motion": "Aggressive movement"}[dominant]
            if not violent:
                label = "Elevated activity"
        return ViolenceResult(score, self._ema, violent, label, persons,
                              {k: round(float(v), 3) for k, v in cues.items()})

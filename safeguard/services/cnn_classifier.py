"""Optional closed-set face classifier: the original FYP custom CNN.

This is a *classifier over a fixed list of 54 identities*, not a face-matching system. It can
only ever answer "which of those 54 people does this face look most like", so on its own it will
confidently mislabel anyone who is not one of them. Two guards make that safer than before:

  * a confidence floor (SAFEGUARD_CNN_MIN_CONF): below it the answer is "unknown", never a guess;
  * no fallback simulation: if TensorFlow or the model cannot load, the backend reports
    unavailable instead of inventing predictions (the original code did the latter).

Loading TensorFlow takes ~40 s, so it happens on a background thread and the rest of the app
starts immediately."""
from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

INPUT_SIZE = 160  # model input; BGR pixels scaled to 0-1, exactly as in training


class ClosedSetCNN:
    def __init__(self, model_dir: Path, min_conf: float = 0.90, enabled: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.min_conf = min_conf
        self.enabled = enabled
        self.model = None
        self.classes: list[str] = []
        self.error: str | None = None if enabled else "disabled (set SAFEGUARD_CNN_ENABLED=true)"
        self._lock = threading.Lock()
        self._loaded = threading.Event()
        if enabled:
            threading.Thread(target=self._load, name="cnn-loader", daemon=True).start()
        else:
            self._loaded.set()

    # ---- loading -----------------------------------------------------------------
    def _load(self) -> None:
        try:
            import keras  # TensorFlow's Keras 3; imported lazily because it is slow and optional
        except Exception as exc:
            self.error = f"TensorFlow/Keras not installed ({exc.__class__.__name__}); pip install -r requirements-cnn.txt"
            self._loaded.set()
            return
        try:
            model_path, enc_path = self.model_dir / "face_recognition_model.h5", self.model_dir / "label_encoder.pkl"
            if not model_path.exists() or not enc_path.exists():
                raise FileNotFoundError(f"missing {model_path.name} or {enc_path.name} in {self.model_dir}")
            with open(enc_path, "rb") as fh:  # the project's own encoder; never load pickles from elsewhere
                encoder = pickle.load(fh)
            self.classes = [str(c) for c in encoder.classes_]
            self.model = keras.models.load_model(str(model_path), compile=False)
            if self.model.output_shape[-1] != len(self.classes):
                raise ValueError("model outputs and label encoder disagree")
            self.error = None
            log.info("closed-set CNN ready: %d identities", len(self.classes))
        except Exception as exc:
            self.model, self.error = None, f"could not load CNN: {exc}"
            log.warning(self.error)
        finally:
            self._loaded.set()

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def state(self) -> str:
        if not self.enabled:
            return "off"
        if not self._loaded.is_set():
            return "loading"
        return "ready" if self.available else "unavailable"

    def wait_ready(self, timeout: float | None = None) -> bool:
        self._loaded.wait(timeout)
        return self.available

    # ---- inference ---------------------------------------------------------------
    @staticmethod
    def crop(frame: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray | None:
        x, y, w, h = box
        px, py = int(w * pad), int(h * pad)
        face = frame[max(y - py, 0):y + h + py, max(x - px, 0):x + w + px]
        return face if face.size else None

    def topk(self, face_bgr: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        if not self.available:
            return []
        x = cv2.resize(face_bgr, (INPUT_SIZE, INPUT_SIZE)).astype("float32")[None] / 255.0
        with self._lock:  # one Keras predict at a time; callers are analysis threads
            probs = self.model.predict(x, verbose=0)[0]
        order = np.argsort(-probs)[:k]
        return [(self.classes[i], float(probs[i])) for i in order]

    def classify(self, face_bgr: np.ndarray) -> tuple[str | None, float]:
        """Best guess, or (None, confidence) when below the confidence floor ('unknown')."""
        top = self.topk(face_bgr, 1)
        if not top:
            return None, 0.0
        label, conf = top[0]
        return (label, conf) if conf >= self.min_conf else (None, conf)

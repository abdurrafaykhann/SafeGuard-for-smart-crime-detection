"""Face detection and recognition.

Backends (all from OpenCV, so no TensorFlow / dlib install):
  * YuNet  ONNX  - detector with 5 landmarks (falls back to a Haar cascade)
  * SFace  ONNX  - 128-d embedding, matched by cosine similarity

The two ONNX files are fetched by `python scripts/download_models.py`. Without them the
system still *detects* faces but reports recognition as unavailable. It never fabricates a
match; the previous version fell back to a random number generator, which is why it
is gone."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    raw: np.ndarray | None = None  # YuNet row (box + landmarks + score) for alignment


@dataclass
class FaceMatch:
    entry_id: int
    label: str
    risk: str
    similarity: float


class FaceEngine:
    def __init__(self, models_dir: Path, match_threshold: float = 0.40) -> None:
        self.match_threshold = match_threshold
        self._lock = threading.Lock()  # cv2 DNN objects are not re-entrant
        self.detector = None
        self.recognizer = None
        self.detector_name = "none"
        self._haar = None
        self._gallery: np.ndarray = np.zeros((0, 128), dtype=np.float32)
        self._gallery_meta: list[tuple[int, str, str]] = []

        yunet, sface = models_dir / YUNET_FILE, models_dir / SFACE_FILE
        try:
            if yunet.exists():
                self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.8, 0.3, 500)
                self.detector_name = "YuNet"
        except Exception as exc:
            log.warning("YuNet failed to load: %s", exc)
        if self.detector is None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.detector_name = "Haar cascade"
        try:
            if sface.exists():
                self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        except Exception as exc:
            log.warning("SFace failed to load: %s", exc)

    @property
    def can_recognize(self) -> bool:
        return self.recognizer is not None

    def status(self) -> dict:
        return {"detector": self.detector_name, "recognition": self.can_recognize,
                "gallery_size": len(self._gallery_meta)}

    # ---- detection ---------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        h, w = frame.shape[:2]
        with self._lock:
            if self.detector is not None:
                self.detector.setInputSize((w, h))
                _, rows = self.detector.detect(frame)
                faces = []
                for r in (rows if rows is not None else []):
                    x, y, bw, bh = (int(v) for v in r[:4])
                    x, y = max(x, 0), max(y, 0)
                    faces.append(DetectedFace((x, y, min(bw, w - x), min(bh, h - y)), float(r[-1]), r))
                return faces
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            found = self._haar.detectMultiScale(gray, 1.15, 6, minSize=(48, 48))
            return [DetectedFace((int(x), int(y), int(bw), int(bh)), 0.9) for (x, y, bw, bh) in found]

    # ---- embeddings --------------------------------------------------------
    def embed(self, frame: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        if not self.can_recognize:
            return None
        with self._lock:
            try:
                if face.raw is not None:
                    aligned = self.recognizer.alignCrop(frame, face.raw)
                else:
                    x, y, w, h = face.box
                    crop = frame[max(y, 0):y + h, max(x, 0):x + w]
                    if crop.size == 0:
                        return None
                    aligned = cv2.resize(crop, (112, 112))
                feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
            except cv2.error as exc:
                log.debug("embedding failed: %s", exc)
                return None
        norm = np.linalg.norm(feat)
        return feat / norm if norm else None

    def largest_face(self, frame: np.ndarray) -> DetectedFace | None:
        faces = self.detect(frame)
        return max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None

    # ---- gallery -----------------------------------------------------------
    def load_gallery(self, entries) -> None:
        vectors, meta = [], []
        for e in entries:
            for v in e.vectors():  # every enrolled photo of a person is a separate row; ranking keeps the best per person
                if v.shape[0] == 128:
                    vectors.append(v)
                    meta.append((e.id, e.label, e.risk))
        self._gallery = np.vstack(vectors) if vectors else np.zeros((0, 128), dtype=np.float32)
        self._gallery_meta = meta

    def _best_per_person(self, embedding: np.ndarray) -> list[tuple[float, int]]:
        """(similarity, gallery row) for each person's closest photo, strongest first."""
        sims = self._gallery @ embedding  # both L2-normalised -> cosine similarity
        best: dict[int, int] = {}
        for i, (entry_id, _, _) in enumerate(self._gallery_meta):
            if entry_id not in best or sims[i] > sims[best[entry_id]]:
                best[entry_id] = i
        return sorted(((float(sims[i]), i) for i in best.values()), reverse=True)

    def match(self, embedding: np.ndarray | None) -> FaceMatch | None:
        if embedding is None or not len(self._gallery_meta):
            return None
        sim, i = self._best_per_person(embedding)[0]
        if sim < self.match_threshold:
            return None
        entry_id, label, risk = self._gallery_meta[i]
        return FaceMatch(entry_id, label, risk, sim)

    def top_matches(self, embedding: np.ndarray, k: int = 5) -> list[FaceMatch]:
        if embedding is None or not len(self._gallery_meta):
            return []
        return [FaceMatch(self._gallery_meta[i][0], self._gallery_meta[i][1], self._gallery_meta[i][2], sim)
                for sim, i in self._best_per_person(embedding)[:k]]

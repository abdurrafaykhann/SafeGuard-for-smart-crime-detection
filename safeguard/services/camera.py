"""Threaded video capture.

The original app opened a fresh `cv2.VideoCapture` for every browser request, so two
viewers meant two cameras fighting over one device. Here each source is captured exactly
once by a background thread that always keeps only the newest frame (no buffer lag), and
any number of consumers read from it."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

log = logging.getLogger(__name__)

# RTSP over TCP avoids the smeared/green frames UDP packet loss causes, and a 5 s open
# timeout makes a dead camera fail fast instead of hanging for 30 s.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000")

ALLOWED_SCHEMES = {"rtsp", "rtsps", "http", "https"}


class InvalidSource(ValueError):
    pass


def validate_source(source: str) -> str:
    """Accept a webcam index or a streaming URL. Local file paths are rejected on purpose:
    a camera 'source' typed into a web form must never be able to read arbitrary files."""
    source = (source or "").strip()
    if source.isdigit() and int(source) < 16:
        return source
    parsed = urlparse(source)
    if parsed.scheme.lower() in ALLOWED_SCHEMES and parsed.netloc:
        return source
    raise InvalidSource("Source must be a webcam index (0-15) or an rtsp/http(s) stream URL.")


class CameraStream:
    def __init__(self, source: str, width: int = 640) -> None:
        self.source = validate_source(source)
        self.width = width
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.fps = 0.0
        self.last_error: str | None = None

    # ---- lifecycle
    def start(self) -> "CameraStream":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.source}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=3)
        self._release()

    def _open(self) -> bool:
        if self.source.isdigit():
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.source), backend)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        return True

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.connected = False

    # ---- capture loop
    def _run(self) -> None:
        backoff = 1.0
        ticks, window_start = 0, time.monotonic()
        while not self._stop.is_set():
            if self._cap is None and not self._open():
                self.connected = False
                self.last_error = "Unable to open video source"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.last_error = "Stream interrupted, reconnecting"
                self._release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            backoff, self.connected, self.last_error = 1.0, True, None
            h, w = frame.shape[:2]
            if w > self.width:
                frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            ticks += 1
            now = time.monotonic()
            if now - window_start >= 2.0:
                self.fps = ticks / (now - window_start)
                ticks, window_start = 0, now

    # ---- consumers
    def read(self) -> tuple[np.ndarray | None, int]:
        with self._cond:
            return (None if self._frame is None else self._frame.copy()), self._frame_id

    def wait_for_frame(self, last_id: int, timeout: float = 2.0) -> tuple[np.ndarray | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or self._stop.is_set(), timeout=timeout)
            return (None if self._frame is None else self._frame.copy()), self._frame_id

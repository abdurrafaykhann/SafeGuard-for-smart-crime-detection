"""Evidence clips: a few seconds of video before and after each alert.

Each camera keeps a short rolling buffer of annotated frames, stored as small JPEGs so five seconds of
footage costs about a megabyte. When an alert fires the buffer is copied (the "before"), recording carries
on for a few more seconds (the "after"), and the result is encoded to WebM (VP8, plays in every current
browser) on a background thread and attached to the alert."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MAX_WIDTH = 640
JPEG_QUALITY = 70


class ClipRecorder:
    def __init__(self, app, pre_s: float, post_s: float, fps: float) -> None:
        self.app = app
        self.pre_s, self.post_s = float(pre_s), float(post_s)
        self.fps = max(float(fps), 1.0)
        self._buffer: deque[tuple[float, bytes]] = deque(maxlen=max(1, int(self.pre_s * self.fps) + 1))
        self._jobs: list[dict] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.pre_s + self.post_s > 0

    @staticmethod
    def _shrink(frame: np.ndarray) -> bytes | None:
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def push(self, frame: np.ndarray, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic() if now is None else now
        jpg = self._shrink(frame)
        if jpg is None:
            return
        done = []
        with self._lock:
            self._buffer.append((now, jpg))
            for job in self._jobs:
                job["frames"].append(jpg)
                if now >= job["until"]:
                    done.append(job)
            self._jobs = [j for j in self._jobs if j not in done]
        for job in done:
            threading.Thread(target=self._finish, args=(job,), name="clip-encode", daemon=True).start()

    def request(self, alert_id: int, now: float | None = None) -> None:
        """Start a clip for this alert: everything buffered so far plus `post_s` more seconds."""
        if not self.enabled:
            return
        now = time.monotonic() if now is None else now
        with self._lock:
            self._jobs.append({"alert_id": alert_id, "frames": [j for _, j in self._buffer], "until": now + self.post_s})

    def flush(self) -> None:
        """Finish pending clips with what exists (camera stopping)."""
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job in jobs:
            self._finish(job)

    def _finish(self, job: dict) -> None:
        from ..extensions import db
        from ..models import Alert
        frames = job["frames"]
        if len(frames) < 2:
            return
        name = f"clip_{job['alert_id']}.webm"
        path: Path = self.app.config["DATA_DIR"] / "clips" / name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), self.fps, (w, h))
            if not writer.isOpened():
                log.warning("could not open a WebM writer; no clip for alert %s", job["alert_id"])
                return
            for jpg in frames:
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] == (h, w):
                    writer.write(img)
            writer.release()
            with self.app.app_context():
                alert = db.session.get(Alert, job["alert_id"])
                if alert is None:  # purged while we were encoding
                    path.unlink(missing_ok=True)
                    return
                alert.clip_path = name
                db.session.commit()
        except Exception:
            log.exception("clip encoding failed for alert %s", job["alert_id"])

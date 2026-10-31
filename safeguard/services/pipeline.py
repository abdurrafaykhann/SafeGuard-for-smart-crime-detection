"""Per-camera analysis pipeline.

One CameraStream (capture thread) feeds one CameraPipeline (analysis thread) which writes
an annotated JPEG that every MJPEG viewer simply reads. Detection cost is therefore paid
once per camera, regardless of how many browsers are watching."""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .alerts import create_alert
from .camera import CameraStream
from .events import bus
from .faces import FaceEngine
from .violence import ViolenceDetector, WeaponBackend, YoloBackend

log = logging.getLogger(__name__)

RED, GREEN, CYAN, AMBER = (60, 60, 255), (120, 230, 130), (255, 220, 90), (60, 190, 255)


class CameraPipeline:
    def __init__(self, app, camera, yolo: YoloBackend, faces: FaceEngine, weapons: WeaponBackend | None = None) -> None:
        self.app, self.faces = app, faces
        # plain snapshot: ORM rows are bound to the request's session and expire in a worker thread
        self.camera = SimpleNamespace(id=camera.id, name=camera.name, source=camera.source,
                                      detect_violence=camera.detect_violence, detect_faces=camera.detect_faces)
        camera = self.camera
        cfg = app.config
        self.stream = CameraStream(camera.source, cfg["FRAME_WIDTH"])
        self.detector = ViolenceDetector(cfg["VIOLENCE_THRESHOLD"], cfg["VIOLENCE_MIN_FRAMES"], yolo, weapons) if camera.detect_violence else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._jpeg_id = 0
        self._cond = threading.Condition()
        self.state = {"violent": False, "score": 0.0, "label": "Clear", "persons": 0, "faces": 0,
                      "matches": [], "fps": 0.0, "connected": False, "cues": {}}
        self._face_marks: list = []
        self._episode_start: float | None = None
        self._episode_escalated = False
        self._last_alert: dict[str, float] = {}

    # ---- lifecycle
    def start(self) -> None:
        self.stream.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"pipe-{self.camera.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
        self.stream.stop()
        with self._cond:
            self._cond.notify_all()

    # ---- MJPEG access
    def wait_jpeg(self, last_id: int, timeout: float = 3.0) -> tuple[bytes | None, int]:
        with self._cond:
            self._cond.wait_for(lambda: self._jpeg_id != last_id or self._stop.is_set(), timeout=timeout)
            return self._jpeg, self._jpeg_id

    def _publish_jpeg(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._cond:
                self._jpeg, self._jpeg_id = buf.tobytes(), self._jpeg_id + 1
                self._cond.notify_all()

    # ---- helpers
    def _cooldown_ok(self, key: str) -> bool:
        now = time.monotonic()
        if now - self._last_alert.get(key, -1e9) >= self.app.config["ALERT_COOLDOWN_SECONDS"]:
            self._last_alert[key] = now
            return True
        return False

    def _handle_weapons(self, res, frame) -> None:
        if res.weapon_alert and self._cooldown_ok("weapon"):
            kinds = sorted({k for k, _, _ in res.weapons})
            best = max(c for _, c, _ in res.weapons)
            create_alert(self.app, kind="weapon", severity="critical" if res.violent else "high",
                         title=f"Weapon detected: {', '.join(kinds)}", confidence=best,
                         camera=self.camera, frame=frame,
                         meta={"weapons": [{"class": k, "confidence": round(c, 3)} for k, c, _ in res.weapons],
                               "with_violence": res.violent})

    def _handle_violence(self, res, frame) -> None:
        now = time.monotonic()
        if res.violent:
            if self._episode_start is None:
                self._episode_start, self._episode_escalated = now, False
                if self._cooldown_ok("violence"):
                    create_alert(self.app, kind="violence", severity="high", title=res.label,
                                 confidence=res.smoothed, camera=self.camera, frame=frame,
                                 meta={"cues": res.cues, "persons": len(res.persons), "mode": self.detector.mode})
            elif not self._episode_escalated and now - self._episode_start >= self.app.config["PERSISTENT_SECONDS"]:
                self._episode_escalated = True
                create_alert(self.app, kind="persistent", severity="critical",
                             title=f"Sustained incident ({int(now - self._episode_start)}s): {res.label}",
                             confidence=res.smoothed, camera=self.camera, frame=frame,
                             meta={"cues": res.cues, "duration_s": round(now - self._episode_start, 1)})
        elif self._episode_start is not None and res.smoothed < self.detector.threshold * 0.6:
            self._episode_start = None

    def _draw_face_marks(self, overlay) -> None:
        """Faces are only re-detected about once a second; redraw the last result on every
        frame so the boxes stay steady instead of flickering."""
        for (x, y, w, h), colour, text in self._face_marks:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2)
            if text:
                cv2.putText(overlay, text, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    def _handle_faces(self, frame) -> None:
        detected = self.faces.detect(frame)
        matches, marks = [], []
        for face in detected:
            match = None
            if self.faces.can_recognize and len(self.faces._gallery_meta):
                match = self.faces.match(self.faces.embed(frame, face))
            marks.append((face.box, RED if match else CYAN,
                          f"{match.label} {match.similarity:.0%}" if match else ""))
            if match:
                matches.append({"label": match.label, "similarity": round(match.similarity, 3), "risk": match.risk})
                if self._cooldown_ok(f"face-{match.entry_id}"):
                    create_alert(self.app, kind="watchlist", severity="high" if match.risk != "low" else "medium",
                                 title=f"Watchlist match: {match.label}", confidence=match.similarity,
                                 camera=self.camera, frame=frame, watchlist_id=match.entry_id,
                                 meta={"risk": match.risk, "similarity": round(match.similarity, 3)})
        self._face_marks = marks
        self.state["faces"], self.state["matches"] = len(detected), matches

    @staticmethod
    def _hud(overlay, violent: bool, label: str, name: str) -> None:
        h, w = overlay.shape[:2]
        colour = RED if violent else GREEN
        L = 18
        for (cx, cy, dx, dy) in ((4, 4, 1, 1), (w - 5, 4, -1, 1), (4, h - 5, 1, -1), (w - 5, h - 5, -1, -1)):
            cv2.line(overlay, (cx, cy), (cx + dx * L, cy), colour, 2)
            cv2.line(overlay, (cx, cy), (cx, cy + dy * L), colour, 2)
        cv2.putText(overlay, f"{name.upper()}  |  {label.upper()}", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        cv2.putText(overlay, time.strftime("%H:%M:%S"), (w - 80, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    # ---- main loop
    def _run(self) -> None:
        cfg = self.app.config
        interval = 1.0 / max(cfg["ANALYSIS_FPS"], 1.0)
        face_every = cfg["FACE_INTERVAL_SECONDS"]
        last_id, last_face, last_state_push = -1, 0.0, 0.0
        res = None
        while not self._stop.is_set():
            started = time.monotonic()
            frame, last_id = self.stream.wait_for_frame(last_id, timeout=2.0)
            self.state["connected"], self.state["fps"] = self.stream.connected, round(self.stream.fps, 1)
            if frame is None:
                time.sleep(0.2)
                continue
            overlay = frame.copy()
            try:
                if self.detector:
                    res = self.detector.analyze(frame)
                    for (x1, y1, x2, y2) in res.persons:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), AMBER if res.violent else GREEN, 1)
                    for name, conf, (x1, y1, x2, y2) in res.weapons:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), RED, 2)
                        cv2.putText(overlay, f"{name.upper()} {conf:.0%}", (x1, max(y1 - 6, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
                    self._handle_weapons(res, frame)
                    self.state.update(violent=res.violent, score=round(res.smoothed, 3), label=res.label,
                                      persons=len(res.persons), cues=res.cues)
                    self._handle_violence(res, frame)
                if self.camera.detect_faces and started - last_face >= face_every:
                    last_face = started
                    self._handle_faces(frame)
                self._draw_face_marks(overlay)
            except Exception:
                log.exception("pipeline error on camera %s", self.camera.id)
            if res is not None and res.violent:
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1] - 1, overlay.shape[0] - 1), RED, 3)
            self._hud(overlay, bool(res and res.violent), self.state["label"] if self.detector else "monitoring", self.camera.name)
            self._publish_jpeg(overlay)

            if started - last_state_push >= 1.0:
                last_state_push = started
                bus.publish("camera", {"id": self.camera.id, **self.state})
            time.sleep(max(0.0, interval - (time.monotonic() - started)))


class CameraManager:
    def __init__(self, app, yolo: YoloBackend, faces: FaceEngine, weapons: WeaponBackend | None = None) -> None:
        self.app, self.yolo, self.faces, self.weapons = app, yolo, faces, weapons
        self._pipelines: dict[int, CameraPipeline] = {}
        self._lock = threading.Lock()

    def start(self, camera) -> CameraPipeline:
        with self._lock:
            if camera.id in self._pipelines:
                return self._pipelines[camera.id]
            pipe = CameraPipeline(self.app, camera, self.yolo, self.faces, self.weapons)
            pipe.start()
            self._pipelines[camera.id] = pipe
            return pipe

    def stop(self, camera_id: int) -> bool:
        with self._lock:
            pipe = self._pipelines.pop(camera_id, None)
        if pipe:
            pipe.stop()
        return pipe is not None

    def get(self, camera_id: int) -> CameraPipeline | None:
        return self._pipelines.get(camera_id)

    def running_ids(self) -> list[int]:
        return list(self._pipelines)

    def stop_all(self) -> None:
        for cid in list(self._pipelines):
            self.stop(cid)

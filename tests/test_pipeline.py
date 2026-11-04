"""End-to-end pipeline test with a fake camera, so no hardware is needed."""
import threading
import time

import cv2
import numpy as np

from safeguard.extensions import db
from safeguard.models import Alert, Camera
from safeguard.services import pipeline as pipeline_mod
from safeguard.services.events import bus


class FakeStream:
    """Same surface as CameraStream; produces a new random frame every 20 ms."""
    connected, fps, last_error = True, 50.0, None

    def __init__(self, source, width=640):
        self._id, self._rng, self._stop = 0, np.random.default_rng(3), threading.Event()

    def start(self):
        return self

    def stop(self):
        self._stop.set()

    def wait_for_frame(self, last_id, timeout=2.0):
        time.sleep(0.02)
        self._id += 1
        # chaotic frame-to-frame change (reads as violent motion) but smooth within a frame: pure per-pixel noise would be
        # rejected as a corrupted frame by services/framecheck.py
        coarse = self._rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
        return cv2.resize(coarse, (320, 240), interpolation=cv2.INTER_LINEAR), self._id


class StubModel:
    available = True

    def infer(self, frame):
        return 0.95, [(10, 10, 120, 220), (70, 12, 170, 225)]


def test_pipeline_raises_one_alert_streams_it_and_serves_mjpeg(app, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "CameraStream", FakeStream)
    app.config.update(ANALYSIS_FPS=40.0, ALERT_COOLDOWN_SECONDS=60.0, PERSISTENT_SECONDS=999)
    with app.app_context():
        cam = Camera(name="Fake", source="0", detect_faces=False)
        db.session.add(cam)
        db.session.commit()
        cam_id = cam.id

    inbox = bus.subscribe()
    pipe = pipeline_mod.CameraPipeline(app, _cam(app, cam_id), StubModel(), app.extensions["safeguard"].faces)
    pipe.start()
    try:
        deadline = time.time() + 6
        while time.time() < deadline and not pipe.state["violent"]:
            time.sleep(0.05)
        time.sleep(0.4)  # let any (suppressed) duplicates try to fire
        assert pipe.state["violent"] and pipe.state["persons"] == 2
        jpeg, _ = pipe.wait_jpeg(-1, timeout=2)
        assert jpeg and jpeg[:2] == b"\xff\xd8"  # a real JPEG for the MJPEG stream
    finally:
        pipe.stop()

    with app.app_context():
        alerts = Alert.query.all()
        assert len(alerts) == 1, "cooldown must collapse repeated detections into a single alert"
        assert alerts[0].kind == "violence" and alerts[0].camera_name == "Fake" and alerts[0].snapshot_path
    events = []
    while not inbox.empty():
        events.append(inbox.get_nowait()[0])
    assert "alert" in events and "camera" in events
    bus.unsubscribe(inbox)


def _cam(app, cam_id):
    with app.app_context():
        return db.session.get(Camera, cam_id)


class StaticStream(FakeStream):
    """A calm scene: identical frames, so the motion cue stays at zero."""

    def wait_for_frame(self, last_id, timeout=2.0):
        time.sleep(0.02)
        self._id += 1
        return np.full((240, 320, 3), 90, np.uint8), self._id


class QuietModel:
    available = True

    def infer(self, frame):
        return 0.0, []


class KnifeStub:
    available = True

    def infer(self, frame):
        return [("knife", 0.9, (30, 30, 120, 160))]


def test_pipeline_raises_weapon_alert_even_without_violence(app, monkeypatch):
    monkeypatch.setattr(pipeline_mod, "CameraStream", StaticStream)
    app.config.update(ANALYSIS_FPS=40.0, ALERT_COOLDOWN_SECONDS=60.0)
    with app.app_context():
        cam = Camera(name="Gate", source="0", detect_faces=False)
        db.session.add(cam)
        db.session.commit()
        cam_id = cam.id
    pipe = pipeline_mod.CameraPipeline(app, _cam(app, cam_id), QuietModel(), app.extensions["safeguard"].faces, KnifeStub())
    pipe.start()
    try:
        deadline = time.time() + 6
        while time.time() < deadline:
            with app.app_context():
                if Alert.query.filter_by(kind="weapon").count():
                    break
            time.sleep(0.1)
        time.sleep(0.3)
    finally:
        pipe.stop()
    with app.app_context():
        alerts = Alert.query.all()
        assert [a.kind for a in alerts] == ["weapon"], "one weapon alert, no violence alert"
        assert alerts[0].severity == "high" and "knife" in alerts[0].title and alerts[0].snapshot_path

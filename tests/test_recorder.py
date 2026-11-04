import time

import cv2
import numpy as np

from safeguard.extensions import db
from safeguard.models import Alert
from safeguard.services.alerts import create_alert
from safeguard.services.recorder import ClipRecorder


def _frame(i):
    f = np.full((240, 320, 3), 40, np.uint8)
    cv2.circle(f, (20 + i * 10, 120), 18, (0, 0, 255), -1)
    return f


def test_clip_contains_frames_from_before_and_after_the_alert(app, admin):
    payload = create_alert(app, kind="violence", severity="high", title="t", frame=_frame(0))
    rec = ClipRecorder(app, pre_s=1.0, post_s=1.0, fps=8)
    t = 100.0
    for i in range(20):                       # 2.5 s of video at 8 fps; the buffer keeps only the last ~1 s
        rec.push(_frame(i), t + i / 8)
        if i == 11:
            rec.request(payload["id"], t + i / 8)
    path = app.config["DATA_DIR"] / "clips" / f"clip_{payload['id']}.webm"
    for _ in range(100):
        if path.exists() and path.stat().st_size and admin.get(f"/api/alerts/{payload['id']}/clip").status_code == 200:
            break
        time.sleep(0.1)
    resp = admin.get(f"/api/alerts/{payload['id']}/clip")
    assert resp.status_code == 200 and resp.mimetype == "video/webm"
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    assert 14 <= n <= 20                      # ~1 s before (9 buffered) + ~1 s after
    with app.app_context():
        assert db.session.get(Alert, payload["id"]).to_dict()["has_clip"]


def test_disabled_recorder_does_nothing(app):
    rec = ClipRecorder(app, 0, 0, 8)
    rec.push(_frame(0))
    rec.request(1)
    assert not rec.enabled and rec._jobs == [] and len(rec._buffer) == 0


def test_alert_without_clip_returns_404(admin, app):
    payload = create_alert(app, kind="violence", severity="high", title="t", frame=_frame(0))
    assert admin.get(f"/api/alerts/{payload['id']}/clip").status_code == 404

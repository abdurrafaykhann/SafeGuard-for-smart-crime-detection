import pytest
import io

import cv2
import numpy as np

from safeguard.extensions import db
from safeguard.models import Alert, utcnow
from safeguard.services.alerts import create_alert, purge_old
from safeguard.services.violence import ViolenceDetector


def _frame(seed=0, size=(240, 320)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (*size, 3), dtype=np.uint8)


# ---------------------------------------------------------------- detector
def test_detector_stays_quiet_on_static_scene():
    det = ViolenceDetector(threshold=0.62, min_frames=3)
    frame = np.full((240, 320, 3), 90, np.uint8)
    results = [det.analyze(frame) for _ in range(15)]
    assert not any(r.violent for r in results)
    assert results[-1].smoothed < 0.1


def test_detector_needs_sustained_signal_not_a_single_spike():
    det = ViolenceDetector(threshold=0.3, min_frames=3)
    calm = np.full((240, 320, 3), 90, np.uint8)
    for _ in range(6):
        det.analyze(calm)
    spike = det.analyze(_frame(1))  # one chaotic frame
    assert not spike.violent


class _StubModel:
    """Stands in for YOLO: reports a confident violence detection with two overlapping people."""
    available = True

    def infer(self, frame):
        return 0.9, [(10, 10, 110, 200), (60, 12, 160, 205)]


def test_detector_flags_sustained_model_signal_and_explains_it():
    det = ViolenceDetector(threshold=0.62, min_frames=3, backend=_StubModel())
    hits = [det.analyze(_frame(i)) for i in range(10)]
    assert not hits[0].violent  # EMA + streak: never on the very first frame
    assert any(r.violent for r in hits)
    assert set(hits[-1].cues) == {"model", "interaction", "blood", "motion", "weapon"}
    assert det.mode == "model+heuristics"


class _StubWeapons:
    available = True

    def __init__(self, conf):
        self.conf = conf

    def infer(self, frame):
        return [("knife", self.conf, (40, 40, 120, 160))]


def test_weapon_alert_needs_confident_sustained_sightings():
    calm = np.full((240, 320, 3), 90, np.uint8)
    strong = ViolenceDetector(min_frames=3, weapons=_StubWeapons(0.8))
    seen = [strong.analyze(calm).weapon_alert for _ in range(5)]
    assert seen[:2] == [False, False] and seen[-1] is True  # not on the first frames
    weak = ViolenceDetector(min_frames=3, weapons=_StubWeapons(0.30))
    assert not any(weak.analyze(calm).weapon_alert for _ in range(10))  # low confidence never alerts


def test_weapon_backend_degrades_gracefully_without_weights(tmp_path):
    from safeguard.services.violence import WeaponBackend
    wb = WeaponBackend(tmp_path / "missing.pt")
    assert not wb.available and "no weapon weights" in wb.error


def test_pure_motion_alone_never_triggers_a_violence_alert():
    """Design guarantee: waving at the camera / a busy scene is not, by itself, violence."""
    det = ViolenceDetector(threshold=0.62, min_frames=3)
    assert not any(det.analyze(_frame(i)).violent for i in range(30))


def test_detector_mode_is_honest_without_a_model():
    assert ViolenceDetector().mode == "heuristics-only"


# ---------------------------------------------------------------- alerts
def test_create_alert_persists_and_saves_snapshot(app):
    payload = create_alert(app, kind="violence", severity="high", title="Test incident", confidence=0.8,
                           frame=_frame(), meta={"cues": {"motion": 0.9}})
    assert payload["id"] and payload["has_snapshot"]
    with app.app_context():
        a = db.session.get(Alert, payload["id"])
        assert (app.config["DATA_DIR"] / "snapshots" / a.snapshot_path).is_file()


def test_ack_flow_and_listing_filters(admin, app):
    create_alert(app, kind="violence", severity="high", title="One", confidence=0.7)
    create_alert(app, kind="persistent", severity="critical", title="Two", confidence=0.9)
    data = admin.get("/api/alerts?severity=critical").get_json()
    assert data["total"] == 1 and data["items"][0]["title"] == "Two"
    aid = data["items"][0]["id"]
    assert admin.post(f"/api/alerts/{aid}/ack").get_json()["acknowledged_by"] == "admin"
    assert admin.get("/api/alerts?status=open").get_json()["total"] == 1


def test_csv_export_neutralises_formula_injection(admin, app):
    create_alert(app, kind="violence", severity="high", title="=HYPERLINK(\"http://evil\")", confidence=0.5)
    body = admin.get("/api/alerts/export.csv").get_data(as_text=True)
    assert "'=HYPERLINK" in body and ",=HYPERLINK" not in body


def test_analytics_summary_shape(admin, app):
    create_alert(app, kind="watchlist", severity="medium", title="Match", confidence=0.6)
    d = admin.get("/api/analytics/summary?days=7").get_json()
    assert d["total"] == 1 and len(d["series"]) == 7 and len(d["by_hour"]) == 24


def test_retention_purges_old_alerts(app):
    payload = create_alert(app, kind="violence", severity="high", title="Old", frame=_frame())
    with app.app_context():
        from datetime import timedelta
        a = db.session.get(Alert, payload["id"])
        a.created_at = utcnow() - timedelta(days=app.config["RETENTION_DAYS"] + 1)
        path = app.config["DATA_DIR"] / "snapshots" / a.snapshot_path
        db.session.commit()
    assert purge_old(app) == 1 and not path.exists()


# ---------------------------------------------------------------- misc routes
def test_fraud_check(client):
    flagged = client.post("/check", data={"domain": "https://www.FakeBank.com/login"})
    assert b"on the fraud list" in flagged.data
    assert b"No report found" in client.post("/check", data={"domain": "example.org"}).data
    bad = client.post("/check", data={"domain": "<script>alert(1)</script>"}).data
    assert b"look like a domain" in bad and b"<script>alert" not in bad  # input is never echoed unescaped


def test_video_upload_rejects_bad_extension(admin):
    resp = admin.post("/api/analysis", data={"video": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_image_upload_rejects_non_images(admin):
    # recognition models are absent in CI, so this must fail cleanly (not crash) with a clear message
    resp = admin.post("/api/search", data={"image": (io.BytesIO(b"not an image"), "a.jpg")}, content_type="multipart/form-data")
    assert resp.status_code == 400 and "message" in resp.get_json()


def test_watchlist_requires_consent_fields(admin, app):
    app.extensions["safeguard"].faces.recognizer = object()  # pretend models are present
    resp = admin.post("/api/watchlist", data={"label": "X"}, content_type="multipart/form-data")
    assert resp.status_code == 400 and "legal" in resp.get_json()["message"].lower()
    resp = admin.post("/api/watchlist", data={"label": "X", "consent_basis": "staff"}, content_type="multipart/form-data")
    assert resp.status_code == 400 and "confirm" in resp.get_json()["message"].lower()


# ---------------------------------------------------------------- optional closed-set CNN
def test_cnn_backend_is_off_by_default_and_never_invents_predictions(tmp_path):
    from safeguard.services.cnn_classifier import ClosedSetCNN
    off = ClosedSetCNN(tmp_path, enabled=False)
    assert off.state == "off" and not off.available
    assert off.classify(np.zeros((50, 50, 3), np.uint8)) == (None, 0.0)  # no fallback simulation
    missing = ClosedSetCNN(tmp_path, enabled=True)   # enabled, but no model files
    missing.wait_ready(10)
    assert missing.state == "unavailable" and missing.error  # reports why, instead of faking results


def test_cnn_confidence_floor_turns_low_confidence_into_unknown():
    from safeguard.services.cnn_classifier import ClosedSetCNN

    class FakeKeras:
        output_shape = (None, 3)

        def predict(self, x, verbose=0):
            assert x.shape == (1, 160, 160, 3) and x.max() <= 1.0  # same preprocessing as training
            return np.array([[0.05, 0.85, 0.10]])

    cnn = ClosedSetCNN("unused", min_conf=0.90, enabled=False)
    cnn.model, cnn.classes = FakeKeras(), ["a", "b", "c"]
    face = np.full((80, 60, 3), 120, np.uint8)
    assert cnn.topk(face, 2)[0][0] == "b"
    assert cnn.classify(face) == (None, pytest.approx(0.85))     # 0.85 < 0.90 floor -> unknown
    cnn.min_conf = 0.80
    assert cnn.classify(face)[0] == "b"                           # above the floor -> label


def test_system_endpoint_reports_cnn_state(admin):
    assert admin.get("/api/system").get_json()["cnn"]["state"] == "off"

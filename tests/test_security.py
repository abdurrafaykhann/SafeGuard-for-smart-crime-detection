import pytest

from safeguard.extensions import db
from safeguard.models import User

from .conftest import login

PROTECTED_API = ["/api/alerts", "/api/cameras", "/api/watchlist", "/api/system", "/api/analytics/summary",
                 "/api/stream", "/api/analysis", "/api/audit"]
PROTECTED_PAGES = ["/console/", "/console/alerts", "/console/analytics", "/console/watchlist",
                   "/console/search", "/console/analysis", "/console/cameras"]


@pytest.mark.parametrize("path", PROTECTED_API)
def test_api_requires_login(client, path):
    resp = client.get(path)
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "unauthorized"


@pytest.mark.parametrize("path", PROTECTED_PAGES)
def test_console_pages_redirect_to_login(client, path):
    resp = client.get(path)
    assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_public_pages_render(client):
    for path in ["/", "/login", "/privacy", "/terms", "/ethics", "/healthz"]:
        assert client.get(path).status_code == 200, path


def test_bad_credentials_get_generic_error(client):
    resp = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 200 and b"Invalid username or password" in resp.data
    resp = client.post("/login", data={"username": "ghost", "password": "wrong"})
    assert b"Invalid username or password" in resp.data  # same message: no user enumeration


def test_account_locks_after_repeated_failures(client, app):
    for _ in range(5):
        client.post("/login", data={"username": "op", "password": "nope"})
    resp = login(client, "op", "operator-pass")  # correct password, but locked
    assert resp.status_code == 200 and b"locked" in resp.data


def test_open_redirect_is_blocked(client):
    resp = client.post("/login?next=https://evil.example/steal", data={"username": "admin", "password": "admin-pass-123"})
    assert resp.status_code == 302 and "evil.example" not in resp.headers["Location"]


def test_security_headers_present(client):
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    csp = resp.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-eval" not in csp


def test_operator_cannot_do_admin_actions(operator):
    assert operator.post("/api/cameras", json={"name": "x", "source": "0"}).status_code == 403
    assert operator.get("/console/cameras").status_code == 403
    assert operator.post("/api/watchlist", data={}).status_code == 403
    assert operator.get("/api/audit").status_code == 403


def test_passwords_are_hashed(app):
    with app.app_context():
        user = db.session.execute(db.select(User).filter_by(username="admin")).scalar_one()
        assert "admin-pass-123" not in user.password_hash and user.password_hash.startswith(("scrypt", "pbkdf2"))


def test_csrf_enforced_when_enabled(app):
    app.config["WTF_CSRF_ENABLED"] = True
    client = app.test_client()
    resp = client.post("/login", data={"username": "admin", "password": "admin-pass-123"})
    assert resp.status_code == 400  # missing CSRF token


def test_camera_source_rejects_local_files(admin):
    for bad in ["C:/Windows/win.ini", "/etc/passwd", "file:///etc/passwd", "../secret.mp4", "99"]:
        resp = admin.post("/api/cameras", json={"name": "bad", "source": bad})
        assert resp.status_code == 400, bad
    assert admin.post("/api/cameras", json={"name": "ok", "source": "rtsp://10.0.0.5/stream"}).status_code == 201
    assert admin.post("/api/cameras", json={"name": "cam", "source": "0"}).status_code == 201


def test_evidence_route_blocks_traversal(admin):
    assert admin.get("/api/evidence/..%2f..%2fsecret_key").status_code == 404
    assert admin.get("/api/evidence/passwd").status_code == 404

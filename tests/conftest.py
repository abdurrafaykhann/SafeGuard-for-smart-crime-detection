import tempfile
from pathlib import Path

import pytest

from safeguard import create_app
from safeguard.config import TestConfig
from safeguard.extensions import db
from safeguard.models import User


@pytest.fixture()
def app(tmp_path):
    class Cfg(TestConfig):
        INSTANCE_DIR = tmp_path
        DATA_DIR = tmp_path / "data"
        ADMIN_PASSWORD = "admin-pass-123"

    app = create_app(Cfg)
    with app.app_context():
        op = User(username="op", role="operator")
        op.set_password("operator-pass")
        db.session.add(op)
        db.session.commit()
    yield app
    app.extensions["safeguard"].manager.stop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, username="admin", password="admin-pass-123"):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=False)


@pytest.fixture()
def admin(client):
    assert login(client).status_code == 302
    return client


@pytest.fixture()
def operator(client):
    assert login(client, "op", "operator-pass").status_code == 302
    return client

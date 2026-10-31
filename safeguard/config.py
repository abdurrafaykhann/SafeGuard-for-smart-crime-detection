"""Central configuration. Every tunable is overridable through environment variables
(or a local `.env` file) so no secret or threshold lives in source code."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # must run before the Config class body reads os.environ


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"SAFEGUARD_{name}", default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return str(_env(name, str(default))).strip().lower() in {"1", "true", "yes", "on"}


def _load_secret_key(instance_dir: Path) -> str:
    """Use the env key if provided, otherwise generate one once and persist it so that
    sessions survive restarts without a secret ever being committed to git."""
    provided = _env("SECRET_KEY")
    if provided:
        return provided
    instance_dir.mkdir(parents=True, exist_ok=True)
    key_file = instance_dir / "secret_key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    key_file.write_text(key, encoding="utf-8")
    return key


class Config:
    INSTANCE_DIR = Path(_env("INSTANCE_DIR", str(BASE_DIR / "instance")))
    DATA_DIR = Path(_env("DATA_DIR", str(INSTANCE_DIR / "data")))
    MODELS_DIR = Path(_env("MODELS_DIR", str(BASE_DIR / "models")))

    SECRET_KEY = _load_secret_key(INSTANCE_DIR)
    SQLALCHEMY_DATABASE_URI = _env("DATABASE_URL", f"sqlite:///{(INSTANCE_DIR / 'safeguard.db').as_posix()}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- session / cookie hardening
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env_bool("COOKIE_SECURE", False)  # set true behind HTTPS
    PERMANENT_SESSION_LIFETIME = _env_int("SESSION_HOURS", 8) * 3600
    WTF_CSRF_TIME_LIMIT = None  # token lives as long as the session

    MAX_CONTENT_LENGTH = _env_int("MAX_UPLOAD_MB", 200) * 1024 * 1024
    RATELIMIT_STORAGE_URI = _env("RATELIMIT_STORAGE", "memory://")
    RATELIMIT_HEADERS_ENABLED = True

    # --- detection tuning
    VIOLENCE_THRESHOLD = _env_float("VIOLENCE_THRESHOLD", 0.62)
    VIOLENCE_MIN_FRAMES = _env_int("VIOLENCE_MIN_FRAMES", 3)
    PERSISTENT_SECONDS = _env_float("PERSISTENT_SECONDS", 12.0)
    ALERT_COOLDOWN_SECONDS = _env_float("ALERT_COOLDOWN", 20.0)
    ANALYSIS_FPS = _env_float("ANALYSIS_FPS", 6.0)
    FACE_INTERVAL_SECONDS = _env_float("FACE_INTERVAL", 1.0)
    FRAME_WIDTH = _env_int("FRAME_WIDTH", 640)
    FACE_MATCH_THRESHOLD = _env_float("FACE_MATCH_THRESHOLD", 0.40)
    YOLO_WEIGHTS = _env("YOLO_WEIGHTS", "Yolo_nano_weights.pt")

    # --- privacy
    RETENTION_DAYS = _env_int("RETENTION_DAYS", 30)
    AUTOSTART_CAMERAS = _env_bool("AUTOSTART_CAMERAS", False)

    # --- optional outbound notifications (all disabled unless configured)
    WEBHOOK_URL = _env("WEBHOOK_URL")
    TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")
    SMTP_HOST = _env("SMTP_HOST")
    SMTP_PORT = _env_int("SMTP_PORT", 587)
    SMTP_USER = _env("SMTP_USER")
    SMTP_PASSWORD = _env("SMTP_PASSWORD")
    ALERT_EMAIL_TO = _env("ALERT_EMAIL_TO")

    # --- first-run admin bootstrap
    ADMIN_USERNAME = _env("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = _env("ADMIN_PASSWORD")


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False
    SECRET_KEY = "test-secret"
    AUTOSTART_CAMERAS = False

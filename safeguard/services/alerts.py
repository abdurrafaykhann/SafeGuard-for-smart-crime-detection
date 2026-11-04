"""Alert persistence + fan-out (browser SSE, webhook, Telegram, email)."""
from __future__ import annotations

import json
import logging
import smtplib
import threading
import uuid
from datetime import timedelta
from email.message import EmailMessage
from urllib import request as urlrequest

import cv2

from ..extensions import db
from ..models import Alert, utcnow
from .events import bus

log = logging.getLogger(__name__)


def snapshot_dir(app):
    d = app.config["DATA_DIR"] / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_snapshot(app, frame, prefix: str = "alert") -> str | None:
    if frame is None:
        return None
    name = f"{prefix}_{utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.jpg"
    ok = cv2.imwrite(str(snapshot_dir(app) / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return name if ok else None


def create_alert(app, *, kind: str, severity: str, title: str, confidence: float = 0.0, camera=None,
                 frame=None, watchlist_id: int | None = None, meta: dict | None = None) -> dict:
    """Persist an alert, publish it to live dashboards and notify external channels."""
    with app.app_context():
        alert = Alert(
            kind=kind, severity=severity, title=title, confidence=float(confidence),
            camera_id=getattr(camera, "id", None), camera_name=getattr(camera, "name", None),
            watchlist_id=watchlist_id, snapshot_path=save_snapshot(app, frame, kind),
            meta=json.dumps(meta or {}),
        )
        db.session.add(alert)
        db.session.commit()
        payload = alert.to_dict()
    bus.publish("alert", payload)
    threading.Thread(target=_notify_external, args=(app.config.copy(), payload), daemon=True).start()
    return payload


def _notify_external(cfg, alert: dict) -> None:
    text = (f"[SafeGuard] {alert['severity'].upper()} - {alert['title']}"
            f" | camera: {alert.get('camera') or 'n/a'} | confidence {alert['confidence']:.0%}")
    try:
        if cfg.get("WEBHOOK_URL"):
            _post_json(cfg["WEBHOOK_URL"], {"text": text, "alert": alert})
        if cfg.get("TELEGRAM_BOT_TOKEN") and cfg.get("TELEGRAM_CHAT_ID"):
            _post_json(f"https://api.telegram.org/bot{cfg['TELEGRAM_BOT_TOKEN']}/sendMessage",
                       {"chat_id": cfg["TELEGRAM_CHAT_ID"], "text": text})
        if cfg.get("SMTP_HOST") and cfg.get("ALERT_EMAIL_TO"):
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = text[:120], cfg.get("SMTP_USER") or "safeguard@localhost", cfg["ALERT_EMAIL_TO"]
            msg.set_content(text)
            with smtplib.SMTP(cfg["SMTP_HOST"], cfg["SMTP_PORT"], timeout=10) as smtp:
                smtp.starttls()
                if cfg.get("SMTP_USER"):
                    smtp.login(cfg["SMTP_USER"], cfg.get("SMTP_PASSWORD") or "")
                smtp.send_message(msg)
    except Exception as exc:  # a failing integration must never break detection
        log.warning("external notification failed: %s", exc)


def _post_json(url: str, body: dict) -> None:
    req = urlrequest.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    urlrequest.urlopen(req, timeout=8).close()


def purge_old(app) -> int:
    """Enforce the retention policy: delete alerts and their snapshots past RETENTION_DAYS."""
    days = app.config["RETENTION_DAYS"]
    if days <= 0:
        return 0
    cutoff = utcnow() - timedelta(days=days)
    with app.app_context():
        old = Alert.query.filter(Alert.created_at < cutoff).all()
        for a in old:
            if a.snapshot_path:
                (snapshot_dir(app) / a.snapshot_path).unlink(missing_ok=True)
            if a.clip_path:
                (app.config["DATA_DIR"] / "clips" / a.clip_path).unlink(missing_ok=True)
            db.session.delete(a)
        db.session.commit()
        return len(old)

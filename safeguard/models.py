from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(16), nullable=False, default="operator")  # admin | operator
    is_active_user = db.Column(db.Boolean, default=True, nullable=False)
    failed_logins = db.Column(db.Integer, default=0, nullable=False)
    locked_until = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=utcnow)
    last_login = db.Column(db.DateTime)

    @property
    def is_active(self) -> bool:  # Flask-Login hook
        return self.is_active_user

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def set_password(self, raw: str) -> None:
        self.password_hash = generate_password_hash(raw, method="scrypt")

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)

    def is_locked(self) -> bool:
        return bool(self.locked_until and self.locked_until > utcnow())


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    source = db.Column(db.String(300), nullable=False)  # "0" for a webcam, or rtsp(s)/http(s) URL
    location = db.Column(db.String(120), default="")
    detect_violence = db.Column(db.Boolean, default=True, nullable=False)
    detect_faces = db.Column(db.Boolean, default=True, nullable=False)
    autostart = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "location": self.location,
            "source_kind": "webcam" if self.source.isdigit() else self.source.split(":", 1)[0],
            "detect_violence": self.detect_violence, "detect_faces": self.detect_faces,
            "autostart": self.autostart,
        }


class WatchlistEntry(db.Model):
    """A person who has been explicitly enrolled with documented consent / authority."""
    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(40), default="person of interest")
    notes = db.Column(db.String(400), default="")
    risk = db.Column(db.String(12), default="medium")  # low | medium | high
    embedding = db.Column(db.LargeBinary, nullable=False)  # float32, L2-normalised
    photo_path = db.Column(db.String(255))
    consent_basis = db.Column(db.String(200), nullable=False)
    created_by = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=utcnow)
    record = db.Column(db.Text, default="")  # JSON "case file" (aliases, offences, status ...); free-form dict
    photos = db.relationship("WatchlistPhoto", backref="entry", cascade="all, delete-orphan", lazy="selectin")

    def vector(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)

    def vectors(self) -> list[np.ndarray]:
        """One face signature per enrolled photo (the entry's own photo plus any extras)."""
        return [self.vector()] + [p.vector() for p in self.photos]

    def case_file(self) -> dict:
        try:
            data = json.loads(self.record or "{}")
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}

    def to_dict(self, detail: bool = False) -> dict:
        d = {
            "id": self.id, "label": self.label, "category": self.category, "notes": self.notes,
            "risk": self.risk, "consent_basis": self.consent_basis, "created_by": self.created_by,
            "created_at": iso(self.created_at), "has_photo": bool(self.photo_path),
            "photo_count": 1 + len(self.photos),
        }
        if detail:
            d["record"] = self.case_file()
        return d


class WatchlistPhoto(db.Model):
    """Additional enrolled photos of the same person; more photos make recognition more robust."""
    id = db.Column(db.Integer, primary_key=True)
    entry_id = db.Column(db.Integer, db.ForeignKey("watchlist_entry.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding = db.Column(db.LargeBinary, nullable=False)
    photo_path = db.Column(db.String(255))

    def vector(self) -> np.ndarray:
        return np.frombuffer(self.embedding, dtype=np.float32)


class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    kind = db.Column(db.String(24), nullable=False, index=True)  # violence | persistent | watchlist
    severity = db.Column(db.String(12), nullable=False, default="high")  # medium | high | critical
    title = db.Column(db.String(160), nullable=False)
    confidence = db.Column(db.Float, default=0.0)
    camera_id = db.Column(db.Integer, db.ForeignKey("camera.id", ondelete="SET NULL"), index=True)
    camera_name = db.Column(db.String(80))
    watchlist_id = db.Column(db.Integer, db.ForeignKey("watchlist_entry.id", ondelete="SET NULL"))
    snapshot_path = db.Column(db.String(255))
    clip_path = db.Column(db.String(255))  # evidence video (WebM) around the alert
    meta = db.Column(db.Text, default="{}")
    acknowledged_by = db.Column(db.String(64))
    acknowledged_at = db.Column(db.DateTime)

    camera = db.relationship("Camera", lazy="joined")
    watchlist = db.relationship("WatchlistEntry", lazy="joined")

    def meta_dict(self) -> dict:
        try:
            return json.loads(self.meta or "{}")
        except ValueError:
            return {}

    def to_dict(self) -> dict:
        return {
            "id": self.id, "created_at": iso(self.created_at), "kind": self.kind,
            "severity": self.severity, "title": self.title,
            "confidence": round(self.confidence or 0.0, 3),
            "camera_id": self.camera_id, "camera": self.camera_name,
            "watchlist": self.watchlist.label if self.watchlist else None,
            "has_snapshot": bool(self.snapshot_path), "has_clip": bool(self.clip_path), "meta": self.meta_dict(),
            "acknowledged_by": self.acknowledged_by, "acknowledged_at": iso(self.acknowledged_at),
        }


class VideoJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    status = db.Column(db.String(16), default="queued")  # queued | running | done | failed
    progress = db.Column(db.Float, default=0.0)
    result = db.Column(db.Text)
    error = db.Column(db.String(300))
    created_by = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=utcnow)

    def to_dict(self) -> dict:
        try:
            result = json.loads(self.result) if self.result else None
        except ValueError:
            result = None
        return {
            "id": self.id, "filename": self.filename, "status": self.status,
            "progress": round(self.progress or 0.0, 3), "result": result,
            "error": self.error, "created_at": iso(self.created_at),
        }


class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    username = db.Column(db.String(64))
    action = db.Column(db.String(60), nullable=False)
    detail = db.Column(db.String(300), default="")
    ip = db.Column(db.String(64))


def audit(username: str | None, action: str, detail: str = "", ip: str | None = None) -> None:
    db.session.add(AuditLog(username=username, action=action, detail=detail[:300], ip=ip))
    db.session.commit()


class Complaint(db.Model):
    """A complaint filed with law enforcement against a business, website or agency.

    Deliberately holds nothing about the complainant: the public checker only ever shows aggregate
    facts about the accused entity (counts, categories, status, dates)."""
    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(db.String(24), unique=True, nullable=False)
    entity_name = db.Column(db.String(160), nullable=False)
    name_key = db.Column(db.String(160), nullable=False, index=True)  # normalised for matching
    entity_domain = db.Column(db.String(253), index=True)
    entity_type = db.Column(db.String(20), default="business")  # business | website | agency
    category = db.Column(db.String(60), nullable=False)
    summary = db.Column(db.String(400), default="")
    status = db.Column(db.String(20), default="reported")  # reported | under_investigation | confirmed | dismissed
    filed_at = db.Column(db.DateTime, default=utcnow)
    created_by = db.Column(db.String(64))

    ACTIVE = ("reported", "under_investigation", "confirmed")


class PublicReport(db.Model):
    """A tip from the public about a business/website. It has NO effect on the fraud check until a staff
    member approves it (which files a normal Complaint). No contact details are collected."""
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    entity_name = db.Column(db.String(160), nullable=False)
    entity_domain = db.Column(db.String(253))
    category = db.Column(db.String(60), nullable=False)
    description = db.Column(db.String(400), default="")
    status = db.Column(db.String(12), default="pending", index=True)  # pending | approved | rejected
    reviewed_by = db.Column(db.String(64))

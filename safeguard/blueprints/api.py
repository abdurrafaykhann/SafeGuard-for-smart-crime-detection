"""JSON / streaming API consumed by the console. Everything here requires a session."""
from __future__ import annotations

import csv
import io
import re
import time
import uuid
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import (Blueprint, Response, abort, current_app, jsonify, redirect, request, send_file,
                   stream_with_context, url_for)
from flask_login import current_user, login_required

from ..extensions import db, limiter, login_manager
from ..models import Alert, AuditLog, Camera, VideoJob, WatchlistEntry, audit, iso, utcnow
from ..security import admin_required, client_ip
from ..services import video_analysis
from ..services.alerts import snapshot_dir
from ..services.camera import InvalidSource, validate_source
from ..services.events import bus

bp = Blueprint("api", __name__, url_prefix="/api")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
EVIDENCE_NAME = re.compile(r"^[a-z0-9]+_\d{8}_\d{6}_[0-9a-f]{6}\.jpg$")


@login_manager.unauthorized_handler
def _unauthorized():
    if request.path.startswith("/api/"):
        return jsonify(error="unauthorized", message="Sign in required"), 401
    return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))


def sg():
    return current_app.extensions["safeguard"]


def _bad(message: str, code: int = 400):
    return jsonify(error="bad_request", message=message), code


def _read_image(field: str = "image") -> np.ndarray:
    """Decode an uploaded image defensively: size cap, real decode (not trusting the
    extension / mimetype), and later re-encoding so no original bytes are ever stored."""
    f = request.files.get(field)
    if not f:
        abort(400, description="No image uploaded")
    data = f.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        abort(413)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        abort(400, description="File is not a valid image")
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


@bp.errorhandler(400)
def _handle_400(err):
    return _bad(getattr(err, "description", "Bad request"))


# ------------------------------------------------------------------ live stream
@bp.get("/stream")
@login_required
def stream():
    resp = Response(stream_with_context(bus.stream()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@bp.get("/system")
@login_required
def system():
    s = sg()
    return jsonify(
        uptime_s=int(time.time() - s.started),
        violence_mode="model+heuristics" if s.yolo.available else "heuristics-only",
        yolo_error=None if s.yolo.available else s.yolo.error,
        cnn={"state": s.cnn.state, "identities": len(s.cnn.classes), "error": s.cnn.error if s.cnn.enabled and not s.cnn.available and s.cnn.state != "loading" else None},
        clip_model={"available": s.clip.available, "threshold": s.clip.threshold if s.clip.available else None,
                    "error": None if s.clip.available else s.clip.error},
        weapon_detector=s.weapons.available, weapon_error=None if s.weapons.available else s.weapons.error,
        faces=s.faces.status(), running_cameras=s.manager.running_ids(),
        viewers=bus.subscriber_count, retention_days=current_app.config["RETENTION_DAYS"],
        user={"name": current_user.username, "role": current_user.role},
    )


# ---------------------------------------------------------------------- cameras
def _camera_payload(cam: Camera) -> dict:
    pipe = sg().manager.get(cam.id)
    d = cam.to_dict()
    d["running"] = pipe is not None
    d["state"] = pipe.state if pipe else None
    return d


@bp.get("/cameras")
@login_required
def cameras_list():
    return jsonify([_camera_payload(c) for c in Camera.query.order_by(Camera.id)])


@bp.post("/cameras")
@admin_required
def cameras_create():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()[:80]
    if not name:
        return _bad("Camera name is required")
    try:
        source = validate_source(str(body.get("source", "")))
    except InvalidSource as exc:
        return _bad(str(exc))
    cam = Camera(name=name, source=source, location=(body.get("location") or "")[:120],
                 detect_violence=bool(body.get("detect_violence", True)),
                 detect_faces=bool(body.get("detect_faces", True)), autostart=bool(body.get("autostart", False)))
    db.session.add(cam)
    db.session.commit()
    audit(current_user.username, "camera_add", f"{cam.name} ({cam.to_dict()['source_kind']})", client_ip())
    return jsonify(_camera_payload(cam)), 201


@bp.delete("/cameras/<int:cid>")
@admin_required
def cameras_delete(cid: int):
    cam = db.get_or_404(Camera, cid)
    sg().manager.stop(cid)
    db.session.delete(cam)
    db.session.commit()
    audit(current_user.username, "camera_delete", cam.name, client_ip())
    return "", 204


@bp.post("/cameras/<int:cid>/start")
@login_required
def cameras_start(cid: int):
    cam = db.get_or_404(Camera, cid)
    sg().manager.start(cam)
    audit(current_user.username, "camera_start", cam.name, client_ip())
    return jsonify(_camera_payload(cam))


@bp.post("/cameras/<int:cid>/stop")
@login_required
def cameras_stop(cid: int):
    cam = db.get_or_404(Camera, cid)
    sg().manager.stop(cid)
    audit(current_user.username, "camera_stop", cam.name, client_ip())
    return jsonify(_camera_payload(cam))


@bp.get("/cameras/<int:cid>/feed")
@login_required
def cameras_feed(cid: int):
    pipe = sg().manager.get(cid)
    if pipe is None:
        abort(404)

    def generate():
        last = -1
        while True:
            jpeg, last_new = pipe.wait_jpeg(last, timeout=3.0)
            if pipe._stop.is_set():
                return
            if jpeg is None or last_new == last:
                continue
            last = last_new
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------------- alerts
def _alert_query():
    q = Alert.query
    a = request.args
    if a.get("kind") in {"violence", "persistent", "watchlist", "weapon"}:
        q = q.filter(Alert.kind == a["kind"])
    if a.get("severity") in {"medium", "high", "critical"}:
        q = q.filter(Alert.severity == a["severity"])
    if a.get("camera", type=int):
        q = q.filter(Alert.camera_id == a.get("camera", type=int))
    if a.get("status") == "open":
        q = q.filter(Alert.acknowledged_at.is_(None))
    elif a.get("status") == "acknowledged":
        q = q.filter(Alert.acknowledged_at.isnot(None))
    if a.get("q"):
        q = q.filter(Alert.title.ilike(f"%{a['q'][:60]}%"))
    return q.order_by(Alert.created_at.desc())


@bp.get("/alerts")
@login_required
def alerts_list():
    page = max(request.args.get("page", 1, type=int), 1)
    limit = min(max(request.args.get("limit", 25, type=int), 1), 100)
    q = _alert_query()
    total = q.count()
    items = q.offset((page - 1) * limit).limit(limit).all()
    return jsonify(items=[i.to_dict() for i in items], total=total, page=page, pages=max((total + limit - 1) // limit, 1))


@bp.post("/alerts/<int:aid>/ack")
@login_required
def alerts_ack(aid: int):
    alert = db.get_or_404(Alert, aid)
    if not alert.acknowledged_at:
        alert.acknowledged_at, alert.acknowledged_by = utcnow(), current_user.username
        db.session.commit()
        audit(current_user.username, "alert_ack", f"#{aid}", client_ip())
        bus.publish("alert_ack", {"id": aid, "by": current_user.username})
    return jsonify(alert.to_dict())


@bp.get("/alerts/<int:aid>/snapshot")
@login_required
def alerts_snapshot(aid: int):
    alert = db.get_or_404(Alert, aid)
    path = snapshot_dir(current_app) / (alert.snapshot_path or "")
    if not alert.snapshot_path or not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=3600)


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in {"=", "+", "-", "@", "\t", "\r"} else text  # spreadsheet formula injection


@bp.get("/alerts/export.csv")
@login_required
def alerts_export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time_utc", "kind", "severity", "title", "confidence", "camera", "watchlist", "acknowledged_by"])
    for a in _alert_query().limit(5000):
        d = a.to_dict()
        w.writerow([_csv_safe(x) for x in (d["id"], d["created_at"], d["kind"], d["severity"], d["title"],
                                           d["confidence"], d["camera"], d["watchlist"], d["acknowledged_by"])])
    audit(current_user.username, "alerts_export", ip=client_ip())
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=safeguard_alerts.csv"})


# -------------------------------------------------------------------- analytics
@bp.get("/analytics/summary")
@login_required
def analytics_summary():
    days = min(max(request.args.get("days", 14, type=int), 1), 90)
    since = utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    rows = Alert.query.filter(Alert.created_at >= since).all()

    per_day: dict[str, Counter] = defaultdict(Counter)
    by_hour, by_kind, by_camera, by_person = [0] * 24, Counter(), Counter(), Counter()
    ack_delays, open_count = [], 0
    for a in rows:
        per_day[a.created_at.strftime("%Y-%m-%d")][a.kind] += 1
        by_hour[a.created_at.hour] += 1
        by_kind[a.kind] += 1
        by_camera[a.camera_name or "Unknown"] += 1
        if a.watchlist:
            by_person[a.watchlist.label] += 1
        if a.acknowledged_at:
            ack_delays.append((a.acknowledged_at - a.created_at).total_seconds())
        else:
            open_count += 1
    series = []
    for i in range(days):
        d = (since + timedelta(days=i)).strftime("%Y-%m-%d")
        series.append({"date": d, **{k: per_day[d][k] for k in ("violence", "persistent", "weapon", "watchlist")}})
    return jsonify(
        days=days, total=len(rows), open=open_count, series=series, by_hour=by_hour,
        by_kind=dict(by_kind), by_camera=by_camera.most_common(6), top_watchlist=by_person.most_common(5),
        critical=sum(1 for a in rows if a.severity == "critical"),
        mean_ack_seconds=round(sum(ack_delays) / len(ack_delays), 1) if ack_delays else None,
        ack_rate=round(len(ack_delays) / len(rows), 3) if rows else None,
    )


# -------------------------------------------------------------------- watchlist
@bp.get("/watchlist")
@login_required
def watchlist_list():
    return jsonify([e.to_dict() for e in WatchlistEntry.query.order_by(WatchlistEntry.created_at.desc())])


@bp.post("/watchlist")
@admin_required
@limiter.limit("20 per hour")
def watchlist_create():
    s = sg()
    if not s.faces.can_recognize:
        return _bad("Face recognition models are not installed. Run: python scripts/download_models.py")
    label = (request.form.get("label") or "").strip()[:100]
    basis = (request.form.get("consent_basis") or "").strip()[:200]
    if not label or not basis:
        return _bad("A name/label and the legal or consent basis are both required.")
    if request.form.get("consent_confirmed") != "on":
        return _bad("You must confirm you have authority or consent to enrol this person.")
    risk = request.form.get("risk") if request.form.get("risk") in {"low", "medium", "high"} else "medium"
    img = _read_image("photo")
    face = s.faces.largest_face(img)
    if face is None:
        return _bad("No face found in that photo. Use a clear, front-facing image.")
    emb = s.faces.embed(img, face)
    if emb is None:
        return _bad("Could not compute a face signature from that photo.")
    x, y, w, h = face.box  # data minimisation: keep only a tight crop, never the full photo
    pad = int(0.2 * max(w, h))
    crop = img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    name = f"wl_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(current_app.config["DATA_DIR"] / "watchlist" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    entry = WatchlistEntry(label=label, category=(request.form.get("category") or "person of interest")[:40],
                           notes=(request.form.get("notes") or "")[:400], risk=risk,
                           embedding=emb.astype(np.float32).tobytes(), photo_path=name,
                           consent_basis=basis, created_by=current_user.username)
    db.session.add(entry)
    db.session.commit()
    s.faces.load_gallery(WatchlistEntry.query.all())
    audit(current_user.username, "watchlist_add", f"#{entry.id} {label} - {basis}", client_ip())
    return jsonify(entry.to_dict()), 201


@bp.delete("/watchlist/<int:eid>")
@admin_required
def watchlist_delete(eid: int):
    entry = db.get_or_404(WatchlistEntry, eid)
    if entry.photo_path:
        (current_app.config["DATA_DIR"] / "watchlist" / entry.photo_path).unlink(missing_ok=True)
    label = entry.label
    db.session.delete(entry)
    db.session.commit()
    sg().faces.load_gallery(WatchlistEntry.query.all())
    audit(current_user.username, "watchlist_delete", f"#{eid} {label}", client_ip())
    return "", 204


@bp.get("/watchlist/<int:eid>")
@login_required
def watchlist_detail(eid: int):
    return jsonify(db.get_or_404(WatchlistEntry, eid).to_dict(detail=True))


@bp.get("/watchlist/<int:eid>/photo")
@login_required
def watchlist_photo(eid: int):
    entry = db.get_or_404(WatchlistEntry, eid)
    path = current_app.config["DATA_DIR"] / "watchlist" / (entry.photo_path or "")
    if not entry.photo_path or not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=3600)


@bp.post("/search")
@login_required
@limiter.limit("30 per hour")
def search_face():
    """Operator tool: upload a photo and rank the closest watchlist entries."""
    s = sg()
    if not s.faces.can_recognize:
        return _bad("Face recognition models are not installed. Run: python scripts/download_models.py")
    img = _read_image("image")
    face = s.faces.largest_face(img)
    if face is None:
        return _bad("No face found in that image.")
    matches = s.faces.top_matches(s.faces.embed(img, face), 5)
    threshold = current_app.config["FACE_MATCH_THRESHOLD"]
    audit(current_user.username, "face_search", f"{len(matches)} candidates", client_ip())
    closed_set = None
    if s.cnn.available:  # original FYP CNN: a guess over a fixed identity list, shown separately and labelled as such
        crop = s.cnn.crop(img, face.box)
        closed_set = {"min_conf": s.cnn.min_conf, "predictions": [
            {"label": lab, "confidence": round(conf, 3), "above_floor": conf >= s.cnn.min_conf}
            for lab, conf in (s.cnn.topk(crop, 3) if crop is not None else [])]}
    by_id = {e.id: e for e in WatchlistEntry.query.filter(WatchlistEntry.id.in_([m.entry_id for m in matches])).all()}
    return jsonify(threshold=threshold, closed_set=closed_set, matches=[
        {"id": m.entry_id, "label": m.label, "risk": m.risk, "similarity": round(m.similarity, 3),
         "is_match": m.similarity >= threshold,
         # the full case file is only sent for real matches; weak candidates get the summary
         **({"details": by_id[m.entry_id].to_dict(detail=True)} if m.similarity >= threshold and m.entry_id in by_id else {})}
        for m in matches])


# ---------------------------------------------------------------------- video
@bp.post("/analysis")
@login_required
@limiter.limit("10 per hour")
def analysis_create():
    f = request.files.get("video")
    if not f or not f.filename:
        return _bad("No video uploaded")
    ext = Path(f.filename).suffix.lower()
    if ext not in video_analysis.ALLOWED_EXTENSIONS:
        return _bad("Unsupported format. Use mp4, avi, mov, mkv or webm.")
    upload_dir = current_app.config["DATA_DIR"] / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}{ext}"  # never trust the client filename on disk
    f.save(path)
    job = VideoJob(filename=Path(f.filename).name[:200], created_by=current_user.username)
    db.session.add(job)
    db.session.commit()
    audit(current_user.username, "video_upload", job.filename, client_ip())
    video_analysis.start_job(current_app._get_current_object(), job.id, path, sg().yolo, sg().faces, sg().weapons, sg().clip)
    return jsonify(job.to_dict()), 202


@bp.get("/analysis")
@login_required
def analysis_list():
    jobs = VideoJob.query.order_by(VideoJob.created_at.desc()).limit(10).all()
    return jsonify([j.to_dict() for j in jobs])


@bp.get("/analysis/<int:jid>")
@login_required
def analysis_get(jid: int):
    return jsonify(db.get_or_404(VideoJob, jid).to_dict())


@bp.get("/evidence/<name>")
@login_required
def evidence(name: str):
    if not EVIDENCE_NAME.match(name):
        abort(404)
    path = snapshot_dir(current_app) / name
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=3600)


@bp.get("/audit")
@admin_required
def audit_log():
    rows = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(100).all()
    return jsonify([{"time": iso(r.created_at), "user": r.username, "action": r.action,
                     "detail": r.detail, "ip": r.ip} for r in rows])

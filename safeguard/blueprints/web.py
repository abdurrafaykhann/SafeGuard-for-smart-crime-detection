"""Public pages: landing, civilian fraud lookup, legal/ethics, health probe."""
from __future__ import annotations

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from ..extensions import db, limiter
from ..services import complaints

bp = Blueprint("web", __name__)


@bp.get("/")
@limiter.limit("60 per minute")
def index():
    """Home page. No login: anyone can look up a business, website or agency name here."""
    q = (request.args.get("q") or "").strip()
    return render_template("index.html", q=q, result=complaints.lookup(q) if q else None)


@bp.route("/check", methods=["GET", "POST"])
@limiter.limit("30 per minute")
def check():
    """Kept for old links: the checker now lives on the home page."""
    q = request.values.get("q") or request.values.get("domain") or ""
    return redirect(url_for("web.index", q=q.strip()[:200]) + "#check" if q.strip() else url_for("web.index") + "#check", 303)


@bp.get("/api/public/check")
@limiter.limit("30 per minute")
def check_json():
    """Same lookup as JSON, for the home page's instant result."""
    return jsonify(complaints.lookup(request.args.get("q", "")))


@bp.get("/privacy")
def privacy():
    return render_template("legal.html", page="privacy")


@bp.get("/terms")
def terms():
    return render_template("legal.html", page="terms")


@bp.get("/ethics")
def ethics():
    return render_template("legal.html", page="ethics")


@bp.get("/healthz")
@limiter.exempt
def healthz():
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify(status="ok")
    except Exception:
        return jsonify(status="degraded"), 503

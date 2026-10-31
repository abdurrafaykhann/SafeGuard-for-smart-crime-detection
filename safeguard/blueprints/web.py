"""Public pages: landing, civilian fraud lookup, legal/ethics, health probe."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from ..extensions import db, limiter

bp = Blueprint("web", __name__)

_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})(?:[/:?#].*)?$")


@lru_cache(maxsize=1)
def _agencies() -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / "data" / "fraud_agencies.json"
    return json.loads(path.read_text(encoding="utf-8"))


@bp.get("/")
def index():
    return render_template("index.html")


@bp.route("/check", methods=["GET", "POST"])
@limiter.limit("30 per minute")
def check():
    """Civilian tool: is this domain on the known-fraud list?"""
    result = None
    if request.method == "POST":
        raw = (request.form.get("domain") or "").strip().lower()[:253]
        m = _DOMAIN.match(raw)
        if not m:
            result = {"query": raw, "valid": False}
        else:
            domain = m.group(1)
            category = _agencies().get(domain)
            result = {"query": domain, "valid": True, "flagged": category is not None, "category": category}
    return render_template("check.html", result=result)


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

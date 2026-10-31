"""Authenticated console pages. Pages are thin shells: data arrives from the JSON API."""
from __future__ import annotations

from flask import Blueprint, render_template
from flask_login import login_required

from ..security import admin_required

bp = Blueprint("console", __name__, url_prefix="/console")


@bp.get("/")
@login_required
def index():
    return render_template("console/live.html", active="live")


@bp.get("/alerts")
@login_required
def alerts():
    return render_template("console/alerts.html", active="alerts")


@bp.get("/analytics")
@login_required
def analytics():
    return render_template("console/analytics.html", active="analytics")


@bp.get("/watchlist")
@login_required
def watchlist():
    return render_template("console/watchlist.html", active="watchlist")


@bp.get("/search")
@login_required
def search():
    return render_template("console/search.html", active="search")


@bp.get("/analysis")
@login_required
def analysis():
    return render_template("console/analysis.html", active="analysis")


@bp.get("/cameras")
@admin_required
def cameras():
    return render_template("console/cameras.html", active="cameras")

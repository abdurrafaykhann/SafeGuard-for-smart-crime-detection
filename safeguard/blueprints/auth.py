from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user, login_required

from ..extensions import db, limiter
from ..models import User, audit, utcnow
from ..security import client_ip, is_safe_redirect

bp = Blueprint("auth", __name__)

MAX_FAILURES = 5
LOCK_MINUTES = 10


@bp.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("console.index"))
    error = None
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()[:64]
        password = request.form.get("password") or ""
        user = User.query.filter_by(username=username).first()
        if user and user.is_locked():
            error = f"Account locked after repeated failures. Try again in {LOCK_MINUTES} minutes."
        elif user and user.is_active and user.check_password(password):
            user.failed_logins, user.locked_until, user.last_login = 0, None, utcnow()
            db.session.commit()
            login_user(user, remember=False)
            audit(username, "login", ip=client_ip())
            target = request.args.get("next")
            return redirect(target if is_safe_redirect(target) else url_for("console.index"))
        else:
            if user:
                user.failed_logins += 1
                if user.failed_logins >= MAX_FAILURES:
                    user.locked_until = utcnow() + timedelta(minutes=LOCK_MINUTES)
                    user.failed_logins = 0
                db.session.commit()
            audit(username or None, "login_failed", ip=client_ip())
            error = "Invalid username or password."  # identical message: no user enumeration
    return render_template("login.html", error=error)


@bp.post("/logout")
@login_required
def logout():
    audit(current_user.username, "logout", ip=client_ip())
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("web.index"))

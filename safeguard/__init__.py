"""SafeGuard - AI-assisted surveillance console (application factory)."""
from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, jsonify, render_template, request

from .config import Config
from .extensions import csrf, db, limiter, login_manager


def create_app(config: type = Config) -> Flask:
    app = Flask(__name__, instance_path=str(config.INSTANCE_DIR))
    app.config.from_object(config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    for d in (app.config["INSTANCE_DIR"], app.config["DATA_DIR"], app.config["DATA_DIR"] / "watchlist"):
        Path(d).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    from .models import Camera, User  # noqa: F401  (register tables)
    from .services.cnn_classifier import ClosedSetCNN
    from .services.faces import FaceEngine
    from .services.pipeline import CameraManager
    from .services.violence import WeaponBackend, YoloBackend

    yolo = YoloBackend(app.config["MODELS_DIR"], app.config["YOLO_WEIGHTS"], app.config["YOLO_IMGSZ"])
    weapons = WeaponBackend(app.config["MODELS_DIR"] / app.config["WEAPON_WEIGHTS"])
    faces = FaceEngine(app.config["MODELS_DIR"], app.config["FACE_MATCH_THRESHOLD"])
    cnn = ClosedSetCNN(app.config["CNN_MODEL_DIR"], app.config["CNN_MIN_CONF"], app.config["CNN_ENABLED"])
    manager = CameraManager(app, yolo, faces, weapons, cnn)
    app.extensions["safeguard"] = SimpleNamespace(yolo=yolo, weapons=weapons, faces=faces, cnn=cnn, manager=manager,
                                                  started=time.time())

    from .blueprints.api import bp as api_bp
    from .blueprints.auth import bp as auth_bp
    from .blueprints.console import bp as console_bp
    from .blueprints.web import bp as web_bp

    for blueprint in (web_bp, auth_bp, console_bp, api_bp):
        app.register_blueprint(blueprint)

    with app.app_context():
        db.create_all()  # additive only; the old build called drop_all() on every start
        _bootstrap_admin(app)
        _refresh_gallery(app)

    _register_hooks(app)
    _register_errors(app)
    _register_cli(app)

    if not app.config.get("TESTING"):
        _background_tasks(app)
    return app


def _bootstrap_admin(app: Flask) -> None:
    from .models import User
    if User.query.count():
        return
    password = app.config["ADMIN_PASSWORD"]
    generated = password is None
    password = password or secrets.token_urlsafe(12)
    user = User(username=app.config["ADMIN_USERNAME"], role="admin")
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    if generated:
        banner = "=" * 62
        app.logger.warning("\n%s\n FIRST RUN - admin account created\n   username: %s\n   password: %s\n"
                           " Shown once. Set SAFEGUARD_ADMIN_PASSWORD to choose your own.\n%s",
                           banner, user.username, password, banner)


def _refresh_gallery(app: Flask) -> None:
    from .models import WatchlistEntry
    app.extensions["safeguard"].faces.load_gallery(WatchlistEntry.query.all())


def _register_hooks(app: Flask) -> None:
    @app.after_request
    def _security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' https://fonts.googleapis.com; "
            "style-src-attr 'unsafe-inline'; "
            "font-src https://fonts.gstatic.com; script-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if request.path.startswith("/api/") or request.endpoint in {"console.index"}:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def _globals():
        sg = app.extensions["safeguard"]
        return {"sys_mode": "model+heuristics" if sg.yolo.available else "heuristics-only",
                "sys_weapons": sg.weapons.available, "sys_cnn": sg.cnn.state,
                "sys_recognition": sg.faces.can_recognize}


def _register_errors(app: Flask) -> None:
    def respond(code: int, title: str, message: str):
        if request.path.startswith("/api/"):
            return jsonify(error=title, message=message), code
        return render_template("error.html", code=code, title=title, message=message), code

    @app.errorhandler(403)
    def _403(_):
        return respond(403, "Access denied", "Your account does not have permission to do that.")

    @app.errorhandler(404)
    def _404(_):
        return respond(404, "Not found", "That page or resource does not exist.")

    @app.errorhandler(413)
    def _413(_):
        return respond(413, "File too large", "The upload exceeds the configured size limit.")

    @app.errorhandler(429)
    def _429(_):
        return respond(429, "Slow down", "Too many requests. Please wait a moment and try again.")

    @app.errorhandler(500)
    def _500(_):
        return respond(500, "Something went wrong", "The error was logged. Please try again.")


def _register_cli(app: Flask) -> None:
    import click

    @app.cli.command("create-user")
    @click.argument("username")
    @click.option("--role", type=click.Choice(["admin", "operator"]), default="operator")
    @click.password_option()
    def create_user(username, role, password):
        """Create an operator/admin account."""
        from .models import User
        if User.query.filter_by(username=username).first():
            raise click.ClickException("User already exists")
        user = User(username=username, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        click.echo(f"Created {role} '{username}'")

    @app.cli.command("purge")
    def purge():
        """Apply the data-retention policy now."""
        from .services.alerts import purge_old
        click.echo(f"Removed {purge_old(app)} expired alert(s)")


def _background_tasks(app: Flask) -> None:
    import threading

    from .models import Camera
    from .services.alerts import purge_old

    def janitor():
        while True:
            try:
                purge_old(app)
            except Exception:
                app.logger.exception("retention purge failed")
            time.sleep(6 * 3600)

    threading.Thread(target=janitor, name="janitor", daemon=True).start()

    if app.config["AUTOSTART_CAMERAS"]:
        with app.app_context():
            for cam in Camera.query.filter_by(autostart=True).all():
                app.extensions["safeguard"].manager.start(cam)

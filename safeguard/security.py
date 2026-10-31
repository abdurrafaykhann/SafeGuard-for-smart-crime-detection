from __future__ import annotations

from functools import wraps
from urllib.parse import urljoin, urlparse

from flask import abort, request
from flask_login import current_user, login_required


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def is_safe_redirect(target: str | None) -> bool:
    """Only follow `?next=` to a path on this same host (open-redirect protection)."""
    if not target:
        return False
    ref, test = urlparse(request.host_url), urlparse(urljoin(request.host_url, target))
    return test.scheme in {"http", "https"} and ref.netloc == test.netloc


def client_ip() -> str:
    return request.remote_addr or "unknown"

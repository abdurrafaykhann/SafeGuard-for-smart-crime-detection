from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

db = SQLAlchemy()
login_manager = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=["600 per hour"])

login_manager.login_view = "auth.login"
login_manager.login_message = "Sign in to access the operations console."
login_manager.login_message_category = "info"
login_manager.session_protection = "strong"

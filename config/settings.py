"""Django settings, configured from environment variables.

Nothing is required for local development: without DATABASE_URL the app uses
the PGlite development server started by `node devdb/server.mjs`, and without
API keys it uses the offline embedder and answerer.
"""

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def database_from_url(url: str) -> dict:
    """Parses postgres://user:password@host:port/name?sslmode=... into Django settings."""
    parts = urlparse(url)
    if parts.scheme not in {"postgres", "postgresql"}:
        raise ImproperlyConfigured("DATABASE_URL must be a postgres:// URL.")
    options = dict(pair.split("=", 1) for pair in parts.query.split("&") if "=" in pair)
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": unquote(parts.path.lstrip("/")) or "postgres",
        "USER": unquote(parts.username or ""),
        "PASSWORD": unquote(parts.password or ""),
        "HOST": parts.hostname or "",
        "PORT": str(parts.port or ""),
        "OPTIONS": options,
        "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "0")),
    }


DEBUG = env_bool("DJANGO_DEBUG", True)
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is off.")
    SECRET_KEY = "dev-only-insecure-key"

ALLOWED_HOSTS = [h for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h]
CSRF_TRUSTED_ORIGINS = [o for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.postgres",
    "rest_framework",
    "rest_framework.authtoken",
    "knowledge",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": database_from_url(os.environ.get("DATABASE_URL", "postgres://postgres@127.0.0.1:5499/postgres?sslmode=disable"))
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "Africa/Nairobi"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LOGIN_URL = "/admin/login/"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
    "DEFAULT_THROTTLE_RATES": {"ask": os.environ.get("ASK_RATE_LIMIT", "30/hour")},
}

# Uploads are processed inside the request, so keep them small (see README, Limitations).
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_BYTES + 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_BYTES + 1024 * 1024

RAG = {
    # "voyage" (semantic, needs VOYAGE_API_KEY) or "hashing" (offline, lexical).
    "EMBEDDING_PROVIDER": os.environ.get("EMBEDDING_PROVIDER", "voyage" if os.environ.get("VOYAGE_API_KEY") else "hashing"),
    "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL", "voyage-3.5"),
    "EMBEDDING_DIMENSIONS": 1024,
    # "claude" (needs ANTHROPIC_API_KEY or an `ant auth login` profile) or "extractive" (offline).
    "ANSWER_PROVIDER": os.environ.get("ANSWER_PROVIDER", "claude" if os.environ.get("ANTHROPIC_API_KEY") else "extractive"),
    "CLAUDE_MODEL": os.environ.get("CLAUDE_MODEL", "claude-opus-5-5"),
    "CLAUDE_EFFORT": os.environ.get("CLAUDE_EFFORT", "medium"),
    "CHUNK_CHARS": 900,
    "CHUNK_OVERLAP": 150,
    "TOP_K": 6,
    # Cosine similarity below this is treated as "not relevant"; with no relevant
    # chunks the API answers "not found" without calling the model.
    "MIN_SIMILARITY": float(os.environ.get("RAG_MIN_SIMILARITY", "0.2")),
}

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_CONTENT_TYPE_NOSNIFF = True

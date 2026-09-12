"""Environment-backed MySQL configuration for the project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
REQUIRED_MYSQL_VARIABLES = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_DATABASE",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
)


def load_project_environment() -> Path:
    """Load the project-root .env and override stale inherited variables."""
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    return ENV_FILE


def mysql_config() -> dict:
    """Return MySQL Connector/Python settings without exposing credentials."""
    load_project_environment()
    missing = [name for name in REQUIRED_MYSQL_VARIABLES if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required MySQL settings in the project-root .env: "
            + ", ".join(missing)
        )

    try:
        port = int(os.environ["MYSQL_PORT"])
    except ValueError as error:
        raise RuntimeError("MYSQL_PORT must be an integer in the project-root .env") from error

    return {
        "host": os.environ["MYSQL_HOST"],
        "port": port,
        "database": os.environ["MYSQL_DATABASE"],
        "user": os.environ["MYSQL_USER"],
        "password": os.environ["MYSQL_PASSWORD"],
    }
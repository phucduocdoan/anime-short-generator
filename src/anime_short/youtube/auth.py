from __future__ import annotations

import json
from pathlib import Path

from anime_short.config import REPO_ROOT

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
ANALYTICS_READONLY_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
SCOPES = [UPLOAD_SCOPE]


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def get_credentials(
    token_file: str,
    client_secret_file: str,
    headless: bool,
    port: int,
    scopes: list[str] | None = None,
):
    """Return OAuth credentials for the requested YouTube scopes, refreshing when possible."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = resolve_repo_path(token_file)
    client_secret_path = resolve_repo_path(client_secret_file)
    requested_scopes = scopes or SCOPES
    has_requested_scopes = _token_has_scopes(token_path, requested_scopes)
    creds = (
        _load_credentials(token_path, Credentials, requested_scopes)
        if has_requested_scopes
        else None
    )

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds, token_path)
            return creds
        except RefreshError:
            creds = None

    if not client_secret_path.exists():
        raise FileNotFoundError(
            f"client_secret.json not found: {client_secret_path}\n"
            "Download an OAuth client secret from Google Cloud Console and place it there."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), requested_scopes)
    creds = flow.run_local_server(port=port, open_browser=not headless, prompt="consent")
    _save_credentials(creds, token_path)
    return creds


def _load_credentials(token_path: Path, credentials_cls, scopes: list[str]):
    if not token_path.exists():
        return None
    try:
        return credentials_cls.from_authorized_user_file(str(token_path), scopes)
    except Exception:
        return None


def _token_has_scopes(token_path: Path, scopes: list[str]) -> bool:
    if not token_path.exists():
        return False
    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    token_scopes = data.get("scopes") or data.get("scope") or []
    if isinstance(token_scopes, str):
        token_scopes = token_scopes.split()
    return set(scopes).issubset(set(token_scopes))


def _save_credentials(creds, token_path: Path) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")

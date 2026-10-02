"""YouTube credentials, shared by scripts/youtube_auth.py (which creates them) and
pipeline/upload.py (which uses them).

One scope only: youtube.upload. It covers videos.insert and thumbnails.set - the docs for
thumbnails.set list it among the accepted scopes - so the broader `youtube` scope, which
would let this token manage and delete anything on the channel, is not asked for. If
thumbnails.set ever answers 403 insufficientPermissions, that scope is the one-line change
here, followed by one more run of scripts/youtube_auth.py.

The token is the long-lived secret: a refresh token that can upload to the channel. It
lives in secrets/token.json, which .gitignore keeps out of the (public) repository. For CI
the same JSON goes in a GitHub Secret and is passed as the YOUTUBE_TOKEN_JSON environment
variable, which is read first.
"""
import json
from pathlib import Path

from core.llm import get_key

ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_PATH = ROOT / "credentials.json"
TOKEN_PATH = ROOT / "secrets" / "token.json"
TOKEN_ENV = "YOUTUBE_TOKEN_JSON"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

REAUTH_HINT = ("run  python scripts/youtube_auth.py  again to sign in and create a fresh "
               "token")


class AuthError(Exception):
    pass


def load_credentials(token_path=TOKEN_PATH):
    """The saved credentials, refreshed if the access token has expired.

    The refresh token itself does not expire while the Google Cloud project's consent
    screen is "In production" (and the token is used at least once every six months). On a
    project left in "Testing" it dies after seven days, which shows up here as a refresh
    failure, and is the first thing to check.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    from_env = get_key(TOKEN_ENV)
    token_path = Path(token_path)
    if from_env:
        info = json.loads(from_env)
    elif token_path.exists():
        info = json.loads(token_path.read_text(encoding="utf-8"))
    else:
        raise AuthError(f"no YouTube token at {token_path} and no {TOKEN_ENV} set; "
                        f"{REAUTH_HINT}")

    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if not creds.refresh_token:
        raise AuthError(f"the saved token has no refresh token; {REAUTH_HINT}")
    if not creds.valid:
        try:
            creds.refresh(Request())
        except RefreshError as e:
            raise AuthError(
                f"Google refused to refresh the YouTube token ({e}). The usual cause is a "
                "consent screen still in 'Testing', which expires tokens after 7 days; "
                f"otherwise it was revoked. Fix that, then {REAUTH_HINT}.") from e
        if not from_env:
            token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def build_service(creds):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=creds, cache_discovery=False)

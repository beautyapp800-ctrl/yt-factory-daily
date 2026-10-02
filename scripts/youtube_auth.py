"""Sign in to YouTube once and save the token the upload stage uses.

    python scripts/youtube_auth.py

Needs credentials.json (the OAuth client downloaded from Google Cloud Console, see
docs/youtube-setup.md) in the project root. A browser window opens; choose the Google
account that owns the channel, press Allow, and this script saves secrets/token.json. After
that nothing here has to be repeated: the saved refresh token renews itself.

It asks for exactly one permission, youtube.upload ("upload videos"), and nothing else.

Two things it does that are easy to miss:
  - access_type=offline and prompt=consent make Google hand over a refresh token every time.
    Without them a repeat sign-in returns only a short-lived access token, and the saved
    file stops working an hour later.
  - it checks the result: the refresh token must be present, and is exchanged once for an
    access token to prove it works. That costs no YouTube API quota.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import youtube


def main():
    if not youtube.CREDENTIALS_PATH.exists():
        print(f"Cannot find {youtube.CREDENTIALS_PATH}.\n"
              "Download the OAuth client JSON from Google Cloud Console, rename it to "
              "credentials.json and put it in the project folder (docs/youtube-setup.md).")
        return 1

    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(youtube.CREDENTIALS_PATH),
                                                     youtube.SCOPES)
    print("A browser window will open. Choose the account that owns the YouTube channel and "
          "press Allow.\nIf Google says it has not verified the app, that is expected for "
          "your own project: Advanced -> Go to the app.\n")
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent",
                                  authorization_prompt_message="",
                                  success_message="Done - you can close this tab and go "
                                                  "back to the terminal.")

    if not creds.refresh_token:
        print("Google did not return a refresh token, so this sign-in would stop working "
              "within an hour. Go to https://myaccount.google.com/permissions, remove this "
              "app, and run the script again.")
        return 1
    granted = set(creds.scopes or [])
    if not set(youtube.SCOPES) <= granted:
        print(f"The permission was not granted (got {sorted(granted)}). Run it again and "
              "leave the upload checkbox ticked.")
        return 1

    creds.refresh(Request())          # proves the refresh token works; no API quota used

    youtube.TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    youtube.TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"\nSaved {youtube.TOKEN_PATH}. It stays on this computer and is excluded from "
          "git; do not send it to anyone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

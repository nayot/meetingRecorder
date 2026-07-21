#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-api-python-client>=2.100",
#   "google-auth-oauthlib>=1.1",
#   "google-auth-httplib2>=0.4",
# ]
# ///
"""
upload — Rename meeting recordings from record.py by matching Google Calendar
events, then move them to a shared Drive folder.

Usage:
    uv run upload.py                       # process every *.m4a in ~/Documents
    uv run upload.py meeting_20260721_143000.m4a
    uv run upload.py --dry-run             # preview without renaming/uploading/deleting
    uv run upload.py --calendar-id you@example.com

For each file:
  1. If the filename matches record.py's default pattern
     (meeting_YYYYMMDD_HHMMSS.m4a), look up the Google Calendar event whose
     time overlaps the recording. If several events match, you're prompted
     to choose. The file is renamed to "<Meeting Title> - <date> <time>.m4a".
     Files that already have a custom name are left as-is.
  2. The (possibly renamed) file is uploaded to the shared Drive folder and
     the local copy is deleted once the upload succeeds.

Auth: reuses ~/.config/meetingRecorder/credentials.json (same OAuth client
as record.py) but caches its own token at
~/.config/meetingRecorder/upload_token.json, since this script needs wider
scopes (full Drive access to write into a pre-existing shared folder, plus
read-only Calendar access) than record.py's drive.file-only token.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_DIR = Path("~/.config/meetingRecorder").expanduser()
CREDS_PATH = CONFIG_DIR / "credentials.json"
TOKEN_PATH = CONFIG_DIR / "upload_token.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# https://drive.google.com/drive/u/0/folders/1okAChiEe7ShBG3GYs53Tf1l2HEvhxPVR
DRIVE_FOLDER_ID = "1okAChiEe7ShBG3GYs53Tf1l2HEvhxPVR"

DEFAULT_DIR = Path("~/Documents").expanduser()
DEFAULT_NAME_RE = re.compile(r"^meeting_(\d{8})_(\d{6})\.m4a$")
DEFAULT_PROXIMITY_MINUTES = 30


# --- Google auth --------------------------------------------------------

def get_credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CREDS_PATH.exists():
        sys.exit(
            f"Google credentials not found at:\n  {CREDS_PATH}\n\n"
            "To set up:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create an OAuth 2.0 Client ID (Desktop app)\n"
            "  3. Enable the Google Drive API and Google Calendar API in the same project\n"
            f"  4. Download the JSON and save it to:\n     {CREDS_PATH}"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    return creds


# --- Filename <-> timestamp --------------------------------------------

def parse_default_timestamp(path: Path) -> datetime | None:
    """Return the recording start time for a record.py default-named file, else None."""
    m = DEFAULT_NAME_RE.match(path.name)
    if not m:
        return None
    date_s, time_s = m.groups()
    naive = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
    return naive.astimezone()  # naive is presumed local time; attach local tzinfo


def get_duration_seconds(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def sanitize_filename(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]', "-", title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "meeting"


def rename_for_meeting(path: Path, title: str, ts: datetime) -> Path:
    safe = sanitize_filename(title)
    date_part = ts.strftime("%Y-%m-%d %H%M")
    new_path = path.with_name(f"{safe} - {date_part}{path.suffix}")
    counter = 1
    while new_path.exists() and new_path != path:
        new_path = path.with_name(f"{safe} - {date_part} ({counter}){path.suffix}")
        counter += 1
    path.rename(new_path)
    return new_path


# --- Calendar matching ---------------------------------------------------

def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _event_times(event: dict) -> tuple[datetime | None, datetime | None]:
    start_raw = event.get("start", {}).get("dateTime")
    end_raw = event.get("end", {}).get("dateTime")
    if not start_raw or not end_raw:
        return None, None  # all-day event; not relevant to a recording
    return _parse_iso(start_raw), _parse_iso(end_raw)


def _list_events(service, calendar_id: str, time_min: datetime, time_max: datetime) -> list[dict]:
    events: list[dict] = []
    page_token = None
    while True:
        resp = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            pageToken=page_token,
        ).execute()
        events.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return events


def _overlap_seconds(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return max(0.0, (earliest_end - latest_start).total_seconds())


def find_matching_events(
    service, calendar_id: str, rec_start: datetime, rec_end: datetime, proximity_minutes: int,
) -> list[dict]:
    """Calendar events overlapping the recording, ranked best match first."""
    window_start = rec_start - timedelta(hours=2)
    window_end = rec_end + timedelta(hours=2)
    events = _list_events(service, calendar_id, window_start, window_end)

    timed = []
    for ev in events:
        s, e = _event_times(ev)
        if s is not None:
            timed.append((ev, s, e))

    overlap_candidates = []
    for ev, s, e in timed:
        ov = _overlap_seconds(rec_start, rec_end, s, e)
        if ov > 0:
            overlap_candidates.append((ov, ev))
    if overlap_candidates:
        overlap_candidates.sort(key=lambda pair: -pair[0])
        return [ev for _, ev in overlap_candidates]

    # No time overlap (e.g. duration unknown) — fall back to proximity of start times.
    near = []
    for ev, s, e in timed:
        delta_min = abs((s - rec_start).total_seconds()) / 60.0
        if delta_min <= proximity_minutes:
            near.append((delta_min, ev))
    near.sort(key=lambda pair: pair[0])
    return [ev for _, ev in near]


def choose_event(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    if not sys.stdin.isatty():
        print(
            "  Multiple calendar events matched; non-interactive session — using closest match.",
            file=sys.stderr,
        )
        return candidates[0]

    print("\n  Multiple calendar events matched this recording:")
    for i, ev in enumerate(candidates, 1):
        start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", "?"))
        print(f"    {i}. {ev.get('summary', '(untitled)')}   [{start}]")
    print("    0. None of these — keep original filename")
    while True:
        choice = input("  Choose a number: ").strip()
        if choice in ("", "0"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("  Invalid input, try again.")


# --- Drive upload ----------------------------------------------------------

def upload_and_delete(drive_service, path: Path) -> None:
    from googleapiclient.http import MediaFileUpload

    print(f"  Uploading {path.name} to Drive…", file=sys.stderr)
    media = MediaFileUpload(str(path), mimetype="audio/mp4", resumable=True)
    uploaded = (
        drive_service.files()
        .create(
            body={"name": path.name, "parents": [DRIVE_FOLDER_ID]},
            media_body=media,
            fields="id,webViewLink",
        )
        .execute()
    )
    link = uploaded.get("webViewLink", f"https://drive.google.com/file/d/{uploaded['id']}/view")
    print(f"  Uploaded: {link}", file=sys.stderr)
    path.unlink()
    print(f"  Deleted local file: {path}", file=sys.stderr)


# --- Per-file pipeline -------------------------------------------------

def process_file(
    path: Path,
    calendar_service,
    drive_service,
    calendar_id: str,
    proximity_minutes: int,
    dry_run: bool,
) -> None:
    print(f"\n=== {path.name} ===", file=sys.stderr)

    final_path = path
    ts = parse_default_timestamp(path)
    if ts is not None:
        duration = get_duration_seconds(path)
        rec_start = ts
        rec_end = ts + timedelta(seconds=duration) if duration else ts + timedelta(minutes=1)

        candidates = find_matching_events(
            calendar_service, calendar_id, rec_start, rec_end, proximity_minutes
        )
        chosen = choose_event(candidates)
        if chosen:
            title = chosen.get("summary") or "meeting"
            if dry_run:
                print(f"  [dry-run] would rename to match calendar event: {title!r}", file=sys.stderr)
            else:
                final_path = rename_for_meeting(path, title, ts)
                print(f"  Renamed to: {final_path.name}", file=sys.stderr)
        else:
            print("  No matching calendar event; keeping original filename.", file=sys.stderr)
    else:
        print("  Custom filename detected; skipping rename step.", file=sys.stderr)

    if dry_run:
        print(f"  [dry-run] would upload {final_path.name} to Drive and delete the local file.", file=sys.stderr)
        return

    upload_and_delete(drive_service, final_path)


# --- CLI -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rename recordings by matching Google Calendar events, then move them to Drive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run upload.py                                  # process every *.m4a in ~/Documents
  uv run upload.py meeting_20260721_143000.m4a       # process one file
  uv run upload.py --dry-run                         # preview only
  uv run upload.py --calendar-id you@example.com     # use a non-primary calendar
""",
    )
    parser.add_argument(
        "files", nargs="*", type=Path,
        help="Recording files to process (default: every *.m4a in ~/Documents)",
    )
    parser.add_argument(
        "--calendar-id", default="primary",
        help="Google Calendar ID to search for matching events (default: primary)",
    )
    parser.add_argument(
        "--proximity-minutes", type=int, default=DEFAULT_PROXIMITY_MINUTES,
        help=f"Fallback window (minutes) for matching by start time alone (default: {DEFAULT_PROXIMITY_MINUTES})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview renames/uploads without changing or deleting anything",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.files:
        files = [Path(f).expanduser().resolve() for f in args.files]
    else:
        files = sorted(DEFAULT_DIR.glob("*.m4a"))

    if not files:
        sys.exit(f"No .m4a files given and none found in {DEFAULT_DIR}")

    missing = [f for f in files if not f.exists()]
    for f in missing:
        print(f"Skipping missing file: {f}", file=sys.stderr)
    files = [f for f in files if f.exists()]
    if not files:
        sys.exit("No existing files to process.")

    from googleapiclient.discovery import build

    print("Authenticating with Google…", file=sys.stderr)
    creds = get_credentials()
    drive_service = build("drive", "v3", credentials=creds)
    calendar_service = build("calendar", "v3", credentials=creds)

    for f in files:
        try:
            process_file(
                f, calendar_service, drive_service,
                args.calendar_id, args.proximity_minutes, args.dry_run,
            )
        except Exception as e:
            print(f"  Error processing {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sounddevice>=0.5.1",
#   "soundfile>=0.12",
#   "numpy>=1.24",
#   "scipy>=1.11",
#   "noisereduce>=3.0.3",
#   "pyloudnorm>=0.1.0",
#   "google-api-python-client>=2.100",
#   "google-auth-oauthlib>=1.1",
#   "google-auth-httplib2>=0.4",
# ]
# ///
"""
record — Cross-platform CLI audio recorder with noise suppression.

Usage:
    uv run record.py                        # mic + system audio → ~/Documents
    uv run record.py --mic-only             # microphone only
    uv run record.py --system-only          # system audio only
    uv run record.py -o /tmp/meeting.wav    # custom output path
    uv run record.py --upload               # upload to Google Drive after recording
    uv run record.py --list-devices         # show device table and exit

Stop recording with Ctrl-C.

Platform notes:
  Linux  — system audio via PipeWire/PulseAudio monitor source (parec).
            Requires `pactl` (ships with pipewire-pulse or pulseaudio).
  Mac    — system audio via BlackHole (brew install blackhole-2ch) or Soundflower.
            Create a Multi-Output Device in Audio MIDI Setup first.
  Windows — system audio via WASAPI loopback on the default output device.
"""

import argparse
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


# --- Constants and config ---------------------------------------------------

CONFIG_DIR = Path("~/.config/meetingRecorder").expanduser()
CREDS_PATH = CONFIG_DIR / "credentials.json"
TOKEN_PATH = CONFIG_DIR / "token.json"
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TARGET_LUFS = -16.0


# --- Platform detection -----------------------------------------------------

def get_platform() -> str:
    s = platform.system()
    if s == "Linux":
        return "linux"
    if s == "Darwin":
        return "mac"
    if s == "Windows":
        return "windows"
    sys.exit(f"Unsupported platform: {s}")


# --- Device / source discovery ----------------------------------------------

@dataclass
class SystemSource:
    """Represents where to capture system audio from."""
    device_idx: int | None     # sounddevice device index (Mac/Windows/ALSA)
    parec_name: str | None     # PulseAudio/PipeWire source name (Linux)
    display_name: str
    native_sr: int
    native_channels: int


def _pactl_find_monitor() -> SystemSource | None:
    """Find the default sink's monitor source via pactl."""
    try:
        sink_result = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=3
        )
        default_sink = sink_result.stdout.strip()
        if not default_sink:
            return None
        monitor_name = f"{default_sink}.monitor"

        # Verify the monitor exists and get its spec
        sources_result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=3,
        )
        for line in sources_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            if parts[1] == monitor_name:
                spec = parts[3]  # e.g. "s16le 2ch 48000Hz"
                ch_m = re.search(r"(\d+)ch", spec)
                hz_m = re.search(r"(\d+)Hz", spec)
                ch = int(ch_m.group(1)) if ch_m else 2
                rate = int(hz_m.group(1)) if hz_m else 48000
                return SystemSource(
                    device_idx=None,
                    parec_name=monitor_name,
                    display_name=f"{monitor_name} (via parec)",
                    native_sr=rate,
                    native_channels=ch,
                )

        # Fallback: use any .monitor source if default sink monitor not found
        for line in sources_result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 4 or ".monitor" not in parts[1]:
                continue
            spec = parts[3]
            ch_m = re.search(r"(\d+)ch", spec)
            hz_m = re.search(r"(\d+)Hz", spec)
            ch = int(ch_m.group(1)) if ch_m else 2
            rate = int(hz_m.group(1)) if hz_m else 48000
            return SystemSource(
                device_idx=None,
                parec_name=parts[1],
                display_name=f"{parts[1]} (via parec)",
                native_sr=rate,
                native_channels=ch,
            )
    except Exception:
        pass
    return None


def find_system_audio_source() -> SystemSource | None:
    """
    Returns a SystemSource for capturing what the computer is playing, or None.
    Linux: uses pactl to find the default sink's monitor, captured via parec.
    Mac:   looks for BlackHole/Soundflower in sounddevice device list.
    Windows: uses WASAPI loopback on the default output device.
    """
    plat = get_platform()
    devices = sd.query_devices()

    if plat == "linux":
        # First try sounddevice (works if PortAudio uses PulseAudio backend)
        for i, d in enumerate(devices):
            if ".monitor" in d["name"].lower() and d["max_input_channels"] > 0:
                return SystemSource(
                    device_idx=i,
                    parec_name=None,
                    display_name=d["name"],
                    native_sr=int(d["default_samplerate"]),
                    native_channels=min(2, d["max_input_channels"]),
                )
        # Fall back to pactl + parec
        return _pactl_find_monitor()

    elif plat == "mac":
        loopback_names = ["blackhole", "soundflower", "loopback"]
        for i, d in enumerate(devices):
            if any(n in d["name"].lower() for n in loopback_names) and d["max_input_channels"] > 0:
                return SystemSource(
                    device_idx=i,
                    parec_name=None,
                    display_name=d["name"],
                    native_sr=int(d["default_samplerate"]),
                    native_channels=min(2, d["max_input_channels"]),
                )
        return None

    else:  # windows — WASAPI loopback
        try:
            default_out = sd.query_devices(kind="output")
            for i, d in enumerate(devices):
                if d["name"] == default_out["name"]:
                    ch = min(2, max(1, int(d["max_output_channels"])))
                    return SystemSource(
                        device_idx=i,
                        parec_name=None,
                        display_name=f"{d['name']} (WASAPI loopback)",
                        native_sr=int(d["default_samplerate"]),
                        native_channels=ch,
                    )
        except Exception:
            pass
        return None


def find_mic_device() -> tuple[int | None, str]:
    """Returns (device_index, name) of the default input device."""
    try:
        default_in = sd.query_devices(kind="input")
        for i, d in enumerate(sd.query_devices()):
            if d["name"] == default_in["name"] and d["max_input_channels"] > 0:
                return i, d["name"]
    except Exception:
        pass
    return None, "No default input device found"


def list_devices() -> None:
    devices = sd.query_devices()
    print(f"\n{'Idx':>4}  {'In':>3}  {'Out':>3}  {'Rate':>6}  Name")
    print("-" * 62)
    for i, d in enumerate(devices):
        tag = " *" if ".monitor" in d["name"].lower() else "  "
        print(
            f"{i:>4}{tag} {d['max_input_channels']:>3}  "
            f"{d['max_output_channels']:>3}  "
            f"{int(d['default_samplerate']):>6}  {d['name']}"
        )
    print("\n* = PulseAudio/PipeWire monitor source (system audio on Linux)\n")
    if get_platform() == "linux":
        src = _pactl_find_monitor()
        if src:
            print(f"Linux system audio (parec): {src.parec_name}")
            print(f"  {src.native_channels}ch @ {src.native_sr} Hz")


# --- Recording threads ------------------------------------------------------

def recording_thread_sd(
    device_idx: int,
    channels: int,
    stop_event: threading.Event,
    buffer: list,
    native_sr_out: list,
    is_windows_loopback: bool = False,
) -> None:
    """sounddevice-based recording thread (mic, or system audio on Mac/Windows)."""
    try:
        device_info = sd.query_devices(device_idx)
        native_sr = int(device_info["default_samplerate"])
        native_sr_out.append(native_sr)

        kwargs: dict = dict(
            device=device_idx,
            samplerate=native_sr,
            channels=channels,
            dtype="float32",
        )
        if is_windows_loopback:
            kwargs["extra_settings"] = sd.WasapiSettings(loopback=True)

        def callback(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                print(f"  [audio] {status}", file=sys.stderr)
            buffer.append(indata.copy())

        with sd.InputStream(callback=callback, **kwargs):
            while not stop_event.is_set():
                stop_event.wait(timeout=0.1)

    except Exception as e:
        print(f"  [error] Sounddevice thread failed: {e}", file=sys.stderr)
        stop_event.set()


def recording_thread_parec(
    source_name: str,
    sample_rate: int,
    channels: int,
    stop_event: threading.Event,
    buffer: list,
    native_sr_out: list,
) -> None:
    """parec-based recording for Linux PipeWire/PulseAudio monitor sources.

    Writes to a temp file (parec buffers differently on pipes vs files) and
    loads the data into buffer after recording stops.
    """
    native_sr_out.append(sample_rate)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
            tmp_path = tmp.name

        proc = subprocess.Popen(
            [
                "parec",
                f"--device={source_name}",
                "--file-format=raw",
                f"--rate={sample_rate}",
                f"--channels={channels}",
                "--format=float32le",
                "--latency-msec=50",  # small buffer → flushes data to file promptly
                tmp_path,
            ],
            stderr=subprocess.PIPE,
        )

        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)

        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    except FileNotFoundError:
        print("  [error] 'parec' not found — install pipewire-pulse or pulseaudio", file=sys.stderr)
        stop_event.set()
        return

    finally:
        if tmp_path and os.path.exists(tmp_path):
            frame_bytes = channels * 4
            raw = open(tmp_path, "rb").read()
            os.unlink(tmp_path)
            usable = (len(raw) // frame_bytes) * frame_bytes
            if usable:
                arr = np.frombuffer(raw[:usable], dtype=np.float32).reshape(-1, channels)
                buffer.append(arr)


# --- Post-processing --------------------------------------------------------

def to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr.mean(axis=1).astype(np.float32)
    return arr.flatten().astype(np.float32)


def resample_audio(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return data
    g = gcd(src_sr, dst_sr)
    return resample_poly(data, dst_sr // g, src_sr // g).astype(np.float32)


def mix_tracks(mic: np.ndarray | None, sys_audio: np.ndarray | None) -> np.ndarray:
    if mic is None and sys_audio is None:
        sys.exit("No audio was captured.")
    if mic is None:
        return sys_audio
    if sys_audio is None:
        return mic
    n = max(len(mic), len(sys_audio))
    mic_p = np.pad(mic, (0, n - len(mic)))
    sys_p = np.pad(sys_audio, (0, n - len(sys_audio)))
    return np.clip(mic_p + sys_p, -1.0, 1.0).astype(np.float32)


def post_process(
    mic_buf: list,
    sys_buf: list,
    mic_sr: int,
    sys_sr: int,
    out_sr: int,
    denoise: bool,
    target_lufs: float,
) -> np.ndarray:
    mic_audio: np.ndarray | None = None
    sys_audio: np.ndarray | None = None

    if mic_buf:
        print("Processing mic track…", file=sys.stderr)
        raw = np.concatenate(mic_buf, axis=0)
        mono = to_mono(raw)
        resampled = resample_audio(mono, mic_sr, out_sr)
        if denoise:
            print("Applying noise suppression…", file=sys.stderr)
            resampled = nr.reduce_noise(
                y=resampled,
                sr=out_sr,
                stationary=False,
                prop_decrease=0.75,
            ).astype(np.float32)
        mic_audio = resampled

    if sys_buf:
        print("Processing system audio track…", file=sys.stderr)
        raw = np.concatenate(sys_buf, axis=0)
        sys_audio = resample_audio(to_mono(raw), sys_sr, out_sr)

    mixed = mix_tracks(mic_audio, sys_audio)

    print("Normalizing loudness…", file=sys.stderr)
    audio64 = mixed.astype(np.float64)
    meter = pyln.Meter(out_sr)
    loudness = meter.integrated_loudness(audio64)
    if math.isfinite(loudness):
        normalized = pyln.normalize.loudness(audio64, loudness, target_lufs)
        mixed = np.clip(normalized, -1.0, 1.0).astype(np.float32)

    return mixed


# --- Google Drive -----------------------------------------------------------

def get_gdrive_credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not CREDS_PATH.exists():
        sys.exit(
            f"Google Drive credentials not found at:\n  {CREDS_PATH}\n\n"
            "To set up:\n"
            "  1. Go to https://console.cloud.google.com/apis/credentials\n"
            "  2. Create an OAuth 2.0 Client ID (Desktop app)\n"
            "  3. Enable the Google Drive API in the same project\n"
            f"  4. Download the JSON and save it to:\n     {CREDS_PATH}"
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), GDRIVE_SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), GDRIVE_SCOPES)
        creds = flow.run_local_server(port=0)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    return creds


def upload_to_gdrive(file_path: Path) -> str:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    print("Authenticating with Google Drive…", file=sys.stderr)
    creds = get_gdrive_credentials()
    service = build("drive", "v3", credentials=creds)

    print(f"Uploading {file_path.name}…", file=sys.stderr)
    media = MediaFileUpload(str(file_path), mimetype="audio/wav", resumable=True)
    uploaded = (
        service.files()
        .create(
            body={"name": file_path.name},
            media_body=media,
            fields="id,webViewLink",
        )
        .execute()
    )
    link = uploaded.get(
        "webViewLink",
        f"https://drive.google.com/file/d/{uploaded['id']}/view",
    )
    print(f"Uploaded: {link}", file=sys.stderr)
    return link


# --- CLI --------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record mic and/or system audio with noise suppression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run record.py                          # mic + system, save to ~/Documents
  uv run record.py --mic-only               # microphone only
  uv run record.py --system-only            # system audio only
  uv run record.py -o /tmp/meeting.wav      # custom output path
  uv run record.py --upload                 # record then upload to Google Drive
  uv run record.py --list-devices           # list audio devices and exit
  uv run record.py --mic-device 2           # pick mic by device index
""",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--mic-only", action="store_true", help="Record microphone only")
    src.add_argument("--system-only", action="store_true", help="Record system audio only")

    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output WAV path (default: ~/Documents/meeting_YYYYMMDD_HHMMSS.wav)",
    )
    parser.add_argument(
        "--mic-device", type=int, default=None,
        help="Mic input device index (see --list-devices)",
    )
    parser.add_argument(
        "--system-device", type=int, default=None,
        help="Override system audio device index (see --list-devices)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help=f"Output sample rate Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--no-denoise", action="store_true",
        help="Skip spectral noise suppression on mic track",
    )
    parser.add_argument(
        "--target-lufs", type=float, default=DEFAULT_TARGET_LUFS,
        help=f"Target loudness in LUFS (default: {DEFAULT_TARGET_LUFS})",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload output WAV to Google Drive after recording",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print audio device table and exit",
    )
    return parser


def make_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        p = Path(args.output).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    docs = Path("~/Documents").expanduser()
    docs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return docs / f"meeting_{timestamp}.wav"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


# --- Main -------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    out_path = make_output_path(args)
    plat = get_platform()

    # Resolve mic device
    mic_device_idx: int | None = None
    mic_device_name = ""
    if not args.system_only:
        if args.mic_device is not None:
            mic_device_idx = args.mic_device
            mic_device_name = str(sd.query_devices(args.mic_device)["name"])
        else:
            mic_device_idx, mic_device_name = find_mic_device()
            if mic_device_idx is None:
                sys.exit(f"Mic not found: {mic_device_name}")

    # Resolve system audio source
    sys_source: SystemSource | None = None
    use_system = not args.mic_only
    if use_system:
        if args.system_device is not None:
            # User explicitly specified a sounddevice index
            d = sd.query_devices(args.system_device)
            sys_source = SystemSource(
                device_idx=args.system_device,
                parec_name=None,
                display_name=str(d["name"]),
                native_sr=int(d["default_samplerate"]),
                native_channels=min(2, max(1, int(d["max_input_channels"]))),
            )
        else:
            sys_source = find_system_audio_source()
            if sys_source is None:
                if args.system_only:
                    plat = get_platform()
                    if plat == "linux":
                        sys.exit(
                            "System audio source not found.\n"
                            "Ensure pipewire-pulse (or pulseaudio) is running:\n"
                            "  systemctl --user start pipewire-pulse"
                        )
                    elif plat == "mac":
                        sys.exit(
                            "No loopback device found. Install BlackHole:\n"
                            "  brew install blackhole-2ch\n"
                            "Then create a Multi-Output Device in Audio MIDI Setup."
                        )
                    else:
                        sys.exit("Could not find a WASAPI loopback device.")
                else:
                    print(
                        "Warning: system audio source not found — recording mic only.",
                        file=sys.stderr,
                    )
                    use_system = False

    # Print device summary
    print(file=sys.stderr)
    if mic_device_idx is not None:
        print(f"  Mic:    [{mic_device_idx}] {mic_device_name}", file=sys.stderr)
    if use_system and sys_source:
        print(f"  System: {sys_source.display_name}", file=sys.stderr)
    print(f"  Output: {out_path}", file=sys.stderr)
    print(file=sys.stderr)

    is_win_loopback = (plat == "windows") and use_system and (
        sys_source is not None and sys_source.device_idx is not None
    )

    stop_event = threading.Event()
    mic_buf: list = []
    sys_buf: list = []
    mic_sr_out: list = []
    sys_sr_out: list = []
    threads: list[threading.Thread] = []

    if mic_device_idx is not None:
        d = sd.query_devices(mic_device_idx)
        mic_channels = min(2, max(1, int(d["max_input_channels"])))
        threads.append(threading.Thread(
            target=recording_thread_sd,
            args=(mic_device_idx, mic_channels, stop_event, mic_buf, mic_sr_out, False),
            name="mic-recorder",
            daemon=True,
        ))

    if use_system and sys_source is not None:
        if sys_source.parec_name:
            threads.append(threading.Thread(
                target=recording_thread_parec,
                args=(
                    sys_source.parec_name,
                    sys_source.native_sr,
                    sys_source.native_channels,
                    stop_event,
                    sys_buf,
                    sys_sr_out,
                ),
                name="sys-recorder",
                daemon=True,
            ))
        else:
            threads.append(threading.Thread(
                target=recording_thread_sd,
                args=(
                    sys_source.device_idx,
                    sys_source.native_channels,
                    stop_event,
                    sys_buf,
                    sys_sr_out,
                    is_win_loopback,
                ),
                name="sys-recorder",
                daemon=True,
            ))

    for t in threads:
        t.start()

    print("Recording… Press Ctrl-C to stop.", file=sys.stderr)
    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        print("\nStopping…", file=sys.stderr)
        stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    if not mic_buf and not sys_buf:
        sys.exit("No audio was captured.")

    mic_sr = mic_sr_out[0] if mic_sr_out else args.sample_rate
    sys_sr = sys_sr_out[0] if sys_sr_out else (
        sys_source.native_sr if sys_source else args.sample_rate
    )

    audio = post_process(
        mic_buf=mic_buf,
        sys_buf=sys_buf if use_system else [],
        mic_sr=mic_sr,
        sys_sr=sys_sr,
        out_sr=args.sample_rate,
        denoise=not args.no_denoise and bool(mic_buf),
        target_lufs=args.target_lufs,
    )

    sf.write(str(out_path), audio, args.sample_rate, subtype="PCM_16")
    duration = len(audio) / args.sample_rate
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path}  ({format_duration(duration)}, {size_mb:.1f} MB)", file=sys.stderr)

    if args.upload:
        upload_to_gdrive(out_path)


if __name__ == "__main__":
    main()

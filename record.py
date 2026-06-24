#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sounddevice>=0.5.1",
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
    uv run record.py -o /tmp/meeting.m4a    # custom output path
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
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from math import gcd
from pathlib import Path

# Restore SIGINT to default so `kill -INT` works even when launched via
# `nohup ... &` (which sets SIGINT to SIG_IGN for background processes).
signal.signal(signal.SIGINT, signal.default_int_handler)

import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import sounddevice as sd
from scipy.signal import butter, lfilter, resample_poly, sosfilt


# --- Constants and config ---------------------------------------------------

CONFIG_DIR = Path("~/.config/meetingRecorder").expanduser()
CREDS_PATH = CONFIG_DIR / "credentials.json"
TOKEN_PATH = CONFIG_DIR / "token.json"
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_GATE_THRESHOLD_DB = -65.0
PREMIX_LUFS = -20.0   # Per-track target before mixing — equalizes mic vs system levels
HIGHPASS_HZ = 80.0    # Remove low-frequency rumble from mic


class ProgressBar:
    """Step-level progress bar for post-processing; falls back to plain text when not a TTY."""

    def __init__(self, total: int, width: int = 28) -> None:
        self._total = total
        self._width = width
        self._current = 0
        self._tty = sys.stderr.isatty()
        self._start = time.monotonic()

    def _render(self, pct: int, label: str, end: str = "") -> None:
        elapsed = time.monotonic() - self._start
        if pct > 0:
            eta = f"{elapsed / pct * (100 - pct):.0f}s"
        else:
            eta = "--"
        filled = int(self._width * pct / 100)
        bar = "█" * filled + "░" * (self._width - filled)
        print(f"\r  [{bar}] {pct:3d}%  ETA {eta:<6}  {label:<35}", end=end, file=sys.stderr, flush=True)

    def step(self, label: str) -> None:
        old_pct = int(100 * self._current / self._total)
        self._current += 1
        new_pct = int(100 * self._current / self._total)
        if not self._tty:
            print(f"  {label}", file=sys.stderr)
            return
        for pct in range(old_pct + 1, new_pct + 1):
            self._render(pct, label, "\n" if pct >= 100 else "")
            if pct < new_pct:
                time.sleep(0.01)

# --- Terminal VU display ----------------------------------------------------

_IS_TTY   = sys.stderr.isatty()
_RED      = "\033[31m" if _IS_TTY else ""
_RESET    = "\033[0m"  if _IS_TTY else ""
_EL       = "\033[K"   if _IS_TTY else ""   # Erase to end of line
_VU_WIDTH = 20
_VU_DB_MIN = -60.0


def render_vu_bar(level_linear: float) -> str:
    if not math.isfinite(level_linear):
        level_linear = 0.0
    db = 20.0 * math.log10(max(level_linear, 1e-9))
    ratio = max(0.0, min(1.0, (db - _VU_DB_MIN) / (-_VU_DB_MIN)))
    filled = int(ratio * _VU_WIDTH)
    return "█" * filled + "░" * (_VU_WIDTH - filled)


def clean_audio(audio: np.ndarray, label: str | None = None) -> np.ndarray:
    """Return finite float32 audio; replace NaN/Inf samples with silence."""
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        if label:
            print(f"Warning: non-finite samples in {label}; replacing with silence.", file=sys.stderr)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio.astype(np.float32, copy=False)


def rms_level(audio: np.ndarray) -> float:
    audio = clean_audio(audio)
    if audio.size == 0:
        return 0.0
    level = float(np.sqrt(np.mean(audio ** 2)))
    return level if math.isfinite(level) else 0.0


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
    level_out: list | None = None,
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

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                print(f"  [audio] {status}", file=sys.stderr)
            chunk = clean_audio(indata.copy())
            buffer.append(chunk)
            if level_out is not None:
                level_out[0] = rms_level(chunk)

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
    level_out: list | None = None,
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

        frame_bytes = channels * 4
        last_size = 0
        while not stop_event.is_set():
            if level_out is not None:
                try:
                    size = os.path.getsize(tmp_path)
                    if size > last_size:
                        read_bytes = min(size - last_size, frame_bytes * 512)
                        with open(tmp_path, "rb") as f:
                            f.seek(size - read_bytes)
                            chunk = f.read(read_bytes)
                        usable = (len(chunk) // frame_bytes) * frame_bytes
                        if usable:
                            arr = np.frombuffer(chunk[:usable], dtype=np.float32)
                            level_out[0] = rms_level(arr)
                        last_size = size
                except Exception:
                    pass
            stop_event.wait(timeout=0.1)

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
                buffer.append(clean_audio(arr, "parec capture"))


# --- Post-processing --------------------------------------------------------

def to_mono(arr: np.ndarray) -> np.ndarray:
    arr = clean_audio(arr)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr.mean(axis=1).astype(np.float32)
    return arr.flatten().astype(np.float32)


def resample_audio(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    data = clean_audio(data)
    if src_sr == dst_sr:
        return data
    g = gcd(src_sr, dst_sr)
    return clean_audio(resample_poly(data, dst_sr // g, src_sr // g), "resampled audio")


def mix_tracks(mic: np.ndarray | None, sys_audio: np.ndarray | None) -> np.ndarray:
    if mic is None and sys_audio is None:
        sys.exit("No audio was captured.")
    if mic is None:
        return clean_audio(sys_audio, "system audio")
    if sys_audio is None:
        return clean_audio(mic, "mic audio")
    mic = clean_audio(mic, "mic audio")
    sys_audio = clean_audio(sys_audio, "system audio")
    n = max(len(mic), len(sys_audio))
    mic_p = np.pad(mic, (0, n - len(mic)))
    sys_p = np.pad(sys_audio, (0, n - len(sys_audio)))
    return clean_audio(np.clip(mic_p + sys_p, -1.0, 1.0), "mixed audio")


def high_pass(audio: np.ndarray, sr: int, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """2nd-order Butterworth high-pass — removes DC offset and low-frequency rumble."""
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return clean_audio(sosfilt(sos, clean_audio(audio)), "high-pass filtered audio")


def noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = DEFAULT_GATE_THRESHOLD_DB,
    hold_ms: float = 100.0,
    smooth_ms: float = 15.0,
) -> np.ndarray:
    """Frame-based noise gate. Cuts audio below threshold with hold + smooth transitions."""
    audio = clean_audio(audio)
    if len(audio) < sr // 50:
        return audio

    threshold = 10.0 ** (threshold_db / 20.0)
    frame_size = max(1, int(sr * 0.005))   # 5 ms frames
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return audio.copy()

    # Per-frame RMS → binary gate decision
    frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    gate_open = (rms > threshold).astype(np.float32)

    # Hold: keep gate open for hold_ms after each above-threshold frame (backward-looking max)
    hold_frames = max(1, int(hold_ms / 5.0))
    held = gate_open.copy()
    for k in range(1, min(hold_frames + 1, n_frames)):
        shifted = np.concatenate([np.zeros(k, dtype=np.float32), gate_open[:-k]])
        held = np.maximum(held, shifted)

    # Expand frame-level gain to per-sample, pad tail to match audio length
    gain = np.repeat(held, frame_size)
    if len(gain) < len(audio):
        gain = np.pad(gain, (0, len(audio) - len(gain)), mode="edge")

    # One-pole low-pass on the gain envelope for smooth fades
    tau = smooth_ms / 1000.0
    alpha = 1.0 - np.exp(-1.0 / (sr * tau))
    smoothed = lfilter([alpha], [1.0, alpha - 1.0], gain)

    return (audio * smoothed.astype(np.float32)).astype(np.float32)


def normalize_track(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize a single track to target LUFS (ITU-R BS.1770)."""
    audio = clean_audio(audio)
    if len(audio) < int(sr * 0.5):
        return audio
    if rms_level(audio) < 1e-8:
        return np.zeros_like(audio, dtype=np.float32)
    audio64 = audio.astype(np.float64)
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio64)
    except Exception as e:
        print(f"Warning: loudness analysis failed ({e}); skipping track normalization.", file=sys.stderr)
        return audio
    if not math.isfinite(loudness):
        return audio
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normalized = pyln.normalize.loudness(audio64, loudness, target_lufs)
    return clean_audio(np.clip(normalized, -1.0, 1.0), "normalized track")


def post_process(
    mic_buf: list,
    sys_buf: list,
    mic_sr: int,
    sys_sr: int,
    out_sr: int,
    denoise: bool,
    gate: bool,
    gate_threshold_db: float,
    target_lufs: float,
) -> np.ndarray:
    mic_audio: np.ndarray | None = None
    sys_audio: np.ndarray | None = None

    both = bool(mic_buf) and bool(sys_buf)
    total_steps = (
        (1 if mic_buf else 0)
        + (1 if mic_buf and denoise else 0)
        + (1 if mic_buf and gate else 0)
        + (1 if sys_buf else 0)
        + (1 if both else 0)
        + 1  # output normalization
    )
    print("Post-processing…", file=sys.stderr)
    pb = ProgressBar(total_steps)

    if mic_buf:
        pb.step("Mic: resample + high-pass filter")
        raw = np.concatenate(mic_buf, axis=0)
        mono = to_mono(raw)
        resampled = resample_audio(mono, mic_sr, out_sr)
        # High-pass removes HVAC/fan/handling rumble before downstream processing
        resampled = high_pass(resampled, out_sr)
        if denoise:
            pb.step("Mic: noise suppression")
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    resampled = nr.reduce_noise(
                        y=resampled,
                        sr=out_sr,
                        stationary=False,
                        prop_decrease=0.9,
                    ).astype(np.float32)
            except Exception as e:
                print(f"\nWarning: noise suppression failed ({e}); continuing without it.", file=sys.stderr)
            resampled = clean_audio(resampled, "noise-suppressed mic")
        if gate:
            pb.step(f"Mic: noise gate ({gate_threshold_db:.0f} dB threshold)")
            resampled = noise_gate(resampled, out_sr, threshold_db=gate_threshold_db)
        mic_audio = resampled

    if sys_buf:
        pb.step("System audio: resample")
        raw = np.concatenate(sys_buf, axis=0)
        sys_audio = resample_audio(to_mono(raw), sys_sr, out_sr)

    # Auto-level: when both tracks are present, normalize each to the same pre-mix LUFS
    # so the mic comes up to match the system audio level before mixing.
    if mic_audio is not None and sys_audio is not None:
        pb.step("Level matching (pre-mix LUFS)")
        mic_audio = normalize_track(mic_audio, out_sr, PREMIX_LUFS)
        sys_audio = normalize_track(sys_audio, out_sr, PREMIX_LUFS)

    mixed = mix_tracks(mic_audio, sys_audio)

    pb.step("Output loudness normalization")
    mixed = clean_audio(mixed, "mixed audio")
    if rms_level(mixed) < 1e-8:
        return np.zeros_like(mixed, dtype=np.float32)
    audio64 = mixed.astype(np.float64)
    meter = pyln.Meter(out_sr)
    try:
        loudness = meter.integrated_loudness(audio64)
    except Exception as e:
        print(f"\nWarning: loudness analysis failed ({e}); skipping output normalization.", file=sys.stderr)
        return mixed
    if math.isfinite(loudness):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normalized = pyln.normalize.loudness(audio64, loudness, target_lufs)
        mixed = clean_audio(np.clip(normalized, -1.0, 1.0), "normalized output")

    return mixed


# --- Output encoding --------------------------------------------------------

def write_m4a(audio: np.ndarray, sample_rate: int, out_path: Path) -> None:
    """Encode float32 mono audio as AAC/M4A via ffmpeg (piped — no temp file)."""
    audio = clean_audio(audio, "final output")
    if audio.size == 0:
        sys.exit("No audio was captured.")
    cmd = [
        "ffmpeg", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    result = subprocess.run(cmd, input=audio.tobytes(), capture_output=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg encoding failed:\n{result.stderr.decode()}")


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
    media = MediaFileUpload(str(file_path), mimetype="audio/mp4", resumable=True)
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
  uv run record.py -o /tmp/meeting.m4a      # custom output path
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
        help="Output M4A path (default: ~/Documents/meeting_YYYYMMDD_HHMMSS.m4a)",
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
        "--no-gate", action="store_true",
        help="Skip noise gate on mic track",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=DEFAULT_GATE_THRESHOLD_DB,
        help=f"Noise gate threshold in dB (default: {DEFAULT_GATE_THRESHOLD_DB})",
    )
    parser.add_argument(
        "--target-lufs", type=float, default=DEFAULT_TARGET_LUFS,
        help=f"Target loudness in LUFS (default: {DEFAULT_TARGET_LUFS})",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload output M4A to Google Drive after recording",
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
    return docs / f"meeting_{timestamp}.m4a"


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
    mic_level: list = [0.0]
    sys_level: list = [0.0]
    threads: list[threading.Thread] = []

    if mic_device_idx is not None:
        d = sd.query_devices(mic_device_idx)
        mic_channels = min(2, max(1, int(d["max_input_channels"])))
        threads.append(threading.Thread(
            target=recording_thread_sd,
            args=(mic_device_idx, mic_channels, stop_event, mic_buf, mic_sr_out, False, mic_level),
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
                    sys_level,
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
                    sys_level,
                ),
                name="sys-recorder",
                daemon=True,
            ))

    for t in threads:
        t.start()

    start_time = time.time()
    print("Recording… Press Ctrl-C to stop.", file=sys.stderr)
    try:
        while not stop_event.is_set():
            if _IS_TTY:
                elapsed = time.time() - start_time
                blink = "●" if int(elapsed * 2) % 2 == 0 else "○"
                parts = [f"\r  {_RED}{blink} REC{_RESET}  {format_duration(elapsed)}"]
                if mic_device_idx is not None:
                    parts.append(f"  Mic: {render_vu_bar(mic_level[0])}")
                if use_system and sys_source is not None:
                    parts.append(f"  Sys: {render_vu_bar(sys_level[0])}")
                print("".join(parts) + _EL, end="", file=sys.stderr, flush=True)
            stop_event.wait(timeout=0.1)
    except KeyboardInterrupt:
        if _IS_TTY:
            print(f"\r{_EL}", end="", file=sys.stderr)
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
        gate=not args.no_gate and bool(mic_buf),
        gate_threshold_db=args.gate_threshold,
        target_lufs=args.target_lufs,
    )

    write_m4a(audio, args.sample_rate, out_path)
    duration = len(audio) / args.sample_rate
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path}  ({format_duration(duration)}, {size_mb:.1f} MB)", file=sys.stderr)

    if args.upload:
        upload_to_gdrive(out_path)


if __name__ == "__main__":
    main()

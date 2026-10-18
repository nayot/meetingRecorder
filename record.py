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
    uv run record.py -o ~/talks/meeting.m4a # custom output path
    uv run record.py --upload               # upload to Google Drive after recording
    uv run record.py --list-devices         # show device table and exit

Stop recording with Ctrl-C.

Platform notes:
  Linux  — system audio via PipeWire/PulseAudio monitor source (parec).
            Requires `pactl` (ships with pipewire-pulse or pulseaudio).
  Mac    — system audio via a Core Audio process tap (macOS 14.2+; helper
            mac_tap.swift is compiled on first run with swiftc). Needs the
            terminal app allowed under Privacy & Security → Screen & System
            Audio Recording. Older macOS: BlackHole/Soundflower loopback, with
            output routed through a Multi-Output Device.
  Windows — system audio via WASAPI loopback on the default output device.
"""

import argparse
import contextlib
import hashlib
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
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
from scipy.io import wavfile
from scipy.signal import butter, lfilter, resample_poly, sosfilt


# --- Constants and config ---------------------------------------------------

__version__ = "1.2"

CONFIG_DIR = Path("~/.config/meetingRecorder").expanduser()
CREDS_PATH = CONFIG_DIR / "credentials.json"
TOKEN_PATH = CONFIG_DIR / "token.json"
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_GATE_THRESHOLD_DB = -65.0
PREMIX_LUFS = -20.0   # Per-track target before mixing — equalizes mic vs system levels
HIGHPASS_HZ = 80.0    # Remove low-frequency rumble from mic
SILENCE_PEAK = 1e-4   # System track peak below this is treated as "captured nothing"

MAC_TAP_SRC = Path(__file__).resolve().parent / "mac_tap.swift"
CACHE_DIR = Path("~/.cache/meetingRecorder").expanduser()


class ProgressBar:
    """
    Weighted progress bar with animated fill and dual ETA (step + overall).

    Steps are assigned weights so expensive operations (noise suppression) own a
    proportionally larger slice of the bar.  A background thread animates fill
    within the current step using an asymptotic curve; the step ETA is derived by
    linear extrapolation from the fill position.  Falls back to plain text when
    stderr is not a TTY.
    """

    _BASE_SECS_PER_WEIGHT = 3.0  # initial calibration: seconds per weight unit

    def __init__(self, total_weight: int, width: int = 28) -> None:
        self._total_weight = total_weight
        self._width = width
        self._tty = sys.stderr.isatty()
        self._overall_start = time.monotonic()
        self._lock = threading.Lock()
        self._done_weight = 0
        self._n_done = 0
        self._secs_per_weight = self._BASE_SECS_PER_WEIGHT  # EMA-updated
        # Current step state (written before render thread starts)
        self._step_start = 0.0
        self._step_label = ""
        self._step_old_pct = 0.0
        self._step_new_pct = 0.0
        self._step_weight = 1
        self._manual_frac: float | None = None
        self._stop_render = threading.Event()

    def _pct(self, weight: int) -> float:
        return 100.0 * weight / self._total_weight

    @staticmethod
    def _fmt_eta(secs: float | None) -> str:
        if secs is None or not math.isfinite(secs) or secs < 0:
            return "--"
        secs = int(secs + 0.5)
        return f"{secs // 60}m{secs % 60:02d}s" if secs >= 60 else f"{secs}s"

    def _render(self, pct: float, label: str, step_eta: float | None, overall_eta: float | None, end: str = "") -> None:
        erase = "\033[K" if self._tty else ""
        filled = int(self._width * pct / 100)
        bar = "█" * filled + "░" * (self._width - filled)
        o = self._fmt_eta(overall_eta)
        s = self._fmt_eta(step_eta)
        line = f"  [{bar}] {int(pct):3d}%  ETA {o:<8} step {s:<8} {label}"
        if self._tty:
            line = fit_to_terminal(line)
        print(f"\r{line}{erase}", end=end, file=sys.stderr, flush=True)

    def _render_loop(self) -> None:
        try:
            while True:
                with self._lock:
                    step_elapsed = time.monotonic() - self._step_start
                    old = self._step_old_pct
                    new = self._step_new_pct
                    weight = self._step_weight
                    label = self._step_label
                    spw = self._secs_per_weight
                    manual = self._manual_frac

                step_range = new - old
                # Real progress (e.g. from ffmpeg) takes priority over the timer-based
                # animation — a fake asymptotic curve can only stall or jump on a step
                # whose real duration doesn't match the estimate. Cap just under 1.0 so
                # the step-exit snap still owns the "100% means done" moment.
                cap = 0.99 if manual is not None else 0.95
                if manual is not None:
                    fill_frac = min(cap, manual)
                else:
                    # tau ensures the fill takes roughly the expected step duration to reach 95%
                    tau = max(spw * weight * 0.6, float(weight) * 2.0)
                    fill_frac = min(cap, 1.0 - math.exp(-step_elapsed / tau))
                current_pct = old + fill_frac * step_range

                # Step ETA: linear extrapolation from fill position toward the cap
                if fill_frac > 0.02:
                    step_eta: float | None = step_elapsed * (cap - fill_frac) / fill_frac
                    step_eta = max(0.0, step_eta)
                else:
                    step_eta = None

                total_elapsed = time.monotonic() - self._overall_start
                overall_eta: float | None = (
                    total_elapsed / (current_pct / 100.0) * (1.0 - current_pct / 100.0)
                    if current_pct > 1.0 else None
                )

                self._render(current_pct, label, step_eta, overall_eta)

                if self._stop_render.wait(timeout=0.1):
                    break
        except Exception:
            pass

    def update_progress(self, frac: float) -> None:
        """External override for the current step's fill fraction (e.g. real ffmpeg progress)."""
        with self._lock:
            self._manual_frac = max(0.0, min(1.0, frac))

    @contextlib.contextmanager
    def step(self, label: str, weight: int = 1):
        """Context manager: animate fill while the body runs, snap to step end on exit."""
        old_w = self._done_weight
        old_pct = self._pct(old_w)
        new_pct = self._pct(old_w + weight)

        if not self._tty:
            print(f"  {label}…", file=sys.stderr)
            t0 = time.monotonic()
            try:
                yield
            finally:
                elapsed = time.monotonic() - t0
                with self._lock:
                    self._done_weight += weight
                    self._n_done += 1
                    rate = elapsed / weight
                    alpha = 0.3 if self._n_done > 1 else 1.0
                    self._secs_per_weight = (1 - alpha) * self._secs_per_weight + alpha * rate
            return

        with self._lock:
            self._step_start = time.monotonic()
            self._step_label = label
            self._step_old_pct = old_pct
            self._step_new_pct = new_pct
            self._step_weight = weight
            self._manual_frac = None
        self._stop_render.clear()

        render_thread = threading.Thread(target=self._render_loop, daemon=True, name="pb-render")
        render_thread.start()

        try:
            yield
        finally:
            elapsed = time.monotonic() - self._step_start
            self._stop_render.set()
            render_thread.join(timeout=0.5)

            with self._lock:
                self._done_weight += weight
                self._n_done += 1
                rate = elapsed / weight
                alpha = 0.3 if self._n_done > 1 else 1.0
                self._secs_per_weight = (1 - alpha) * self._secs_per_weight + alpha * rate

            if new_pct >= 100.0:
                self._render(100.0, label, None, None, end="\n")
            else:
                total_elapsed = time.monotonic() - self._overall_start
                overall_eta = (
                    total_elapsed / (new_pct / 100.0) * (1.0 - new_pct / 100.0)
                    if new_pct > 1.0 else None
                )
                self._render(new_pct, label, 0.0, overall_eta, end="")

# --- Terminal VU display ----------------------------------------------------

_IS_TTY   = sys.stderr.isatty()
_RED      = "\033[31m" if _IS_TTY else ""
_RESET    = "\033[0m"  if _IS_TTY else ""
_EL       = "\033[K"   if _IS_TTY else ""   # Erase to end of line
_VU_WIDTH = 20
_ANSI_RE = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def fit_to_terminal(line: str) -> str:
    """Clip a status line to the terminal width, ignoring ANSI codes.

    A line that wraps breaks the \\r redraw: the cursor returns only to the
    start of the last wrapped row, so every refresh scrolls the screen.
    """
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns - 1
    out, visible, pos = [], 0, 0
    for m in _ANSI_RE.finditer(line):
        text = line[pos:m.start()]
        take = max(0, cols - visible)
        out.append(text[:take])
        visible += min(len(text), take)
        out.append(m.group())
        pos = m.end()
    out.append(line[pos:][:max(0, cols - visible)])
    return "".join(out)
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
    flat = audio.reshape(-1)
    sq = sum(
        float(np.dot(b, b)) for b in (flat[i : i + (1 << 20)].astype(np.float64) for i in range(0, flat.size, 1 << 20))
    )
    level = math.sqrt(sq / audio.size)
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
    tap_binary: Path | None = None  # mac_tap helper (macOS Core Audio process tap)


def _mac_supports_tap() -> bool:
    """Core Audio process taps need macOS 14.2+."""
    try:
        parts = [int(x) for x in platform.mac_ver()[0].split(".")]
    except ValueError:
        return False
    return tuple((parts + [0, 0])[:2]) >= (14, 2)


def _mac_tap_binary() -> Path | None:
    """Return the compiled mac_tap helper, building it with swiftc if needed."""
    if not MAC_TAP_SRC.exists():
        return None
    digest = hashlib.sha256(MAC_TAP_SRC.read_bytes()).hexdigest()[:12]
    binary = CACHE_DIR / f"mac_tap-{digest}"
    if binary.exists():
        return binary
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Compiling system-audio helper (first run only)…", file=sys.stderr)
    try:
        result = subprocess.run(
            ["swiftc", "-O", str(MAC_TAP_SRC), "-o", str(binary)],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  [warn] Could not build mac_tap ({e}); install Xcode Command Line Tools.",
              file=sys.stderr)
        return None
    if result.returncode != 0:
        print(f"  [warn] mac_tap build failed:\n{result.stderr.strip()}", file=sys.stderr)
        return None
    return binary


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
    Mac:   Core Audio process tap (macOS 14.2+); else BlackHole/Soundflower.
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
        if _mac_supports_tap():
            binary = _mac_tap_binary()
            if binary:
                return SystemSource(
                    device_idx=None,
                    parec_name=None,
                    display_name="Core Audio process tap (all system output)",
                    native_sr=48000,
                    native_channels=2,
                    tap_binary=binary,
                )
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
    elif get_platform() == "mac":
        if _mac_supports_tap():
            print("Mac system audio: Core Audio process tap (default; no BlackHole needed)")
        else:
            print("Mac system audio: BlackHole/Soundflower (macOS < 14.2, no process tap)")


# --- Capture files ----------------------------------------------------------
#
# Every track is written to a raw float32 file beside the output while recording,
# and the file is kept until the M4A has been written. Nothing lives only in RAM,
# so a crash or OOM kill during post-processing loses no audio: rerun with
# --resume, or encode the raw file by hand with ffmpeg. (Not under /tmp: it is
# usually a tmpfs, so a "temp file" there is memory too.)

_CAPTURE_RE = re.compile(r"\.(mic|sys)\.(\d+)Hz\.(\d+)ch\.f32le\.raw$")


@dataclass
class Capture:
    path: Path
    sample_rate: int
    channels: int

    def load(self) -> np.ndarray | None:
        """Memory-map the capture (pages in from disk on demand, not into RSS)."""
        frame = self.channels * 4
        n = self.path.stat().st_size // frame if self.path.exists() else 0
        if n == 0:
            return None
        return np.memmap(self.path, dtype=np.float32, mode="r", shape=(n, self.channels))


def capture_path(out_path: Path, track: str, sample_rate: int, channels: int) -> Path:
    return out_path.with_name(f"{out_path.stem}.{track}.{sample_rate}Hz.{channels}ch.f32le.raw")


def find_captures(out_path: Path) -> dict[str, Capture]:
    """Capture files left beside out_path by an earlier run, keyed by track."""
    found: dict[str, Capture] = {}
    for p in sorted(out_path.parent.glob(f"{glob_escape(out_path.stem)}.*.raw")):
        m = _CAPTURE_RE.search(p.name)
        if m and p.name[: m.start()] == out_path.stem:
            found[m.group(1)] = Capture(p, int(m.group(2)), int(m.group(3)))
    return found


def glob_escape(name: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", name)


def print_recovery_help(captures: list[Capture], out_path: Path) -> None:
    print("\nRaw capture kept on disk:", file=sys.stderr)
    for c in captures:
        print(f"  {c.path}", file=sys.stderr)
    print(
        f"Finish it with:\n  uv run record.py --resume -o \"{out_path}\"\n"
        f"or encode a track directly, e.g.:",
        file=sys.stderr,
    )
    c = captures[0]
    print(
        f"  ffmpeg -f f32le -ar {c.sample_rate} -ac {c.channels} -i \"{c.path}\" "
        f"-c:a aac -b:a 192k \"{out_path}\"",
        file=sys.stderr,
    )


# --- Recording threads ------------------------------------------------------

def recording_thread_sd(
    device_idx: int,
    channels: int,
    stop_event: threading.Event,
    captures: list,
    out_path: Path,
    track: str,
    is_windows_loopback: bool = False,
    level_out: list | None = None,
) -> None:
    """sounddevice-based recording thread (mic, or system audio on Mac/Windows).

    Each callback's block goes straight to the capture file.
    """
    try:
        device_info = sd.query_devices(device_idx)
        native_sr = int(device_info["default_samplerate"])
        path = capture_path(out_path, track, native_sr, channels)
        captures.append(Capture(path, native_sr, channels))
        f = open(path, "wb")

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
            chunk = clean_audio(indata)
            f.write(chunk.tobytes())
            if level_out is not None:
                level_out[0] = rms_level(chunk)

        with f, sd.InputStream(callback=callback, **kwargs):
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
    captures: list,
    out_path: Path,
    level_out: list | None = None,
) -> None:
    """parec-based recording for Linux PipeWire/PulseAudio monitor sources.

    parec writes straight to the capture file (it buffers differently on pipes
    vs files).
    """
    path = capture_path(out_path, "sys", sample_rate, channels)
    captures.append(Capture(path, sample_rate, channels))
    tmp_path = str(path)
    try:
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


def recording_thread_tap(
    binary: Path,
    stop_event: threading.Event,
    captures: list,
    out_path: Path,
    level_out: list | None = None,
) -> None:
    """macOS system audio via the mac_tap helper (Core Audio process tap).

    Same pattern as recording_thread_parec: the helper writes interleaved
    float32 to the capture file. It prints "FORMAT <rate> <channels>" once
    capture is set up; the file is then renamed to carry that format.
    """
    tmp_path = str(out_path.with_name(f"{out_path.stem}.sys.raw"))
    proc = None
    try:
        proc = subprocess.Popen(
            [str(binary), tmp_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        line = proc.stdout.readline().split()
        if len(line) != 3 or line[0] != "FORMAT":
            err = proc.stderr.read().strip()
            print(f"  [error] System audio tap failed: {err or 'no output'}", file=sys.stderr)
            Path(tmp_path).unlink(missing_ok=True)
            stop_event.set()
            return
        native_sr, channels = int(line[1]), int(line[2])
        path = capture_path(out_path, "sys", native_sr, channels)
        os.replace(tmp_path, path)   # the helper keeps writing through its open fd
        tmp_path = str(path)
        captures.append(Capture(path, native_sr, channels))

        frame_bytes = channels * 4
        last_size = 0
        while not stop_event.is_set():
            if proc.poll() is not None:
                # Exit code 0 = Ctrl-C reached the helper too; not an error.
                if proc.returncode != 0:
                    print(f"  [error] System audio tap exited: {proc.stderr.read().strip()}",
                          file=sys.stderr)
                break
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

    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()


def track_peak(audio: np.ndarray | None, block: int = 1 << 20) -> float:
    """Peak |sample|, a block at a time so a long memmapped track never loads whole."""
    if audio is None:
        return 0.0
    peak = 0.0
    for i in range(0, len(audio), block):
        b = np.nan_to_num(np.asarray(audio[i : i + block]))
        if b.size:
            peak = max(peak, float(np.max(np.abs(b))))
    return peak


def warn_if_silent(sys_cap: "Capture | None", plat: str) -> None:
    """Warn loudly when the system track is empty or digital silence."""
    peak = track_peak(sys_cap.load() if sys_cap else None)
    if peak >= SILENCE_PEAK:
        return
    hint = {
        "mac": (
            "Allow your terminal app under System Settings → Privacy & Security →\n"
            "    Screen & System Audio Recording, then restart the terminal.\n"
            "    (BlackHole only receives sound if output goes to a Multi-Output Device.)"
        ),
        "linux": "Check that the default sink is the one playing (pactl get-default-sink).",
        "windows": "Check that the default output device is the one playing.",
    }[plat]
    print(
        f"\n{_RED}WARNING: system audio track is silent.{_RESET}\n"
        f"    Nothing was captured from the computer's output.\n    {hint}\n",
        file=sys.stderr,
    )


# --- Post-processing --------------------------------------------------------
#
# Whole-track operations would allocate several float64 copies of the track
# (a 2-hour mono track at 44.1 kHz is 2.5 GB per float64 copy). Filters,
# resampling and loudness therefore run a block at a time, carrying filter state
# across blocks, so the results match the whole-track versions while peak memory
# stays near one float32 copy of each track.

BLOCK = 1 << 20   # samples per processing block (~24 s at 44.1 kHz)


def to_mono(arr: np.ndarray) -> np.ndarray:
    arr = clean_audio(arr)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr.mean(axis=1, dtype=np.float32)
    return np.array(arr.reshape(-1), dtype=np.float32)


def resample_audio(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase resampling in overlapping blocks aligned to the resampling ratio."""
    data = clean_audio(data)
    if src_sr == dst_sr:
        return data
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    n = len(data)
    out_len = -(-n * up // down)
    out = np.empty(out_len, dtype=np.float32)
    step = down * max(1, (src_sr * 30) // down)   # ~30 s, a multiple of `down`
    pad = down * max(1, src_sr // down)           # ~1 s either side absorbs edge effects
    for i in range(0, n, step):
        lo, hi = max(0, i - pad), min(n, i + step + pad)
        y = resample_poly(data[lo:hi], up, down)
        start = (i - lo) * up // down
        o = i * up // down
        cnt = min(step * up // down, out_len - o)
        out[o : o + cnt] = y[start : start + cnt]
    return clean_audio(out, "resampled audio")


def mix_tracks(mic: np.ndarray | None, sys_audio: np.ndarray | None) -> np.ndarray:
    if mic is None and sys_audio is None:
        sys.exit("No audio was captured.")
    if mic is None:
        return clean_audio(sys_audio, "system audio")
    if sys_audio is None:
        return clean_audio(mic, "mic audio")
    mic = clean_audio(mic, "mic audio")
    sys_audio = clean_audio(sys_audio, "system audio")
    longer, shorter = (mic, sys_audio) if len(mic) >= len(sys_audio) else (sys_audio, mic)
    out = writable(longer)
    out[: len(shorter)] += shorter
    np.clip(out, -1.0, 1.0, out=out)
    return clean_audio(out, "mixed audio")


def writable(audio: np.ndarray) -> np.ndarray:
    return audio if audio.flags.writeable else audio.copy()


def high_pass(audio: np.ndarray, sr: int, cutoff: float = HIGHPASS_HZ) -> np.ndarray:
    """2nd-order Butterworth high-pass — removes DC offset and low-frequency rumble."""
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    audio = clean_audio(audio)
    out = np.empty_like(audio)
    zi = np.zeros((sos.shape[0], 2))
    for i in range(0, len(audio), BLOCK):
        out[i : i + BLOCK], zi = sosfilt(sos, audio[i : i + BLOCK], zi=zi)
    return clean_audio(out, "high-pass filtered audio")


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
    fb = max(1, BLOCK // frame_size)
    rms = np.concatenate([
        np.sqrt(np.mean(frames[i : i + fb] ** 2, axis=1) + 1e-12) for i in range(0, n_frames, fb)
    ])
    gate_open = (rms > threshold).astype(np.float32)

    # Hold: keep gate open for hold_ms after each above-threshold frame (backward-looking max)
    hold_frames = max(1, int(hold_ms / 5.0))
    held = gate_open.copy()
    for k in range(1, min(hold_frames + 1, n_frames)):
        shifted = np.concatenate([np.zeros(k, dtype=np.float32), gate_open[:-k]])
        held = np.maximum(held, shifted)

    # Expand frame-level gain to per-sample (tail padded with the last frame's
    # gain), then a one-pole low-pass on the envelope for smooth fades — a block
    # at a time, carrying the filter state across blocks.
    tau = smooth_ms / 1000.0
    alpha = 1.0 - np.exp(-1.0 / (sr * tau))
    out = np.empty_like(audio)
    zi = np.zeros(1)
    step = fb * frame_size
    for i in range(0, len(audio), step):
        n = min(step, len(audio) - i)
        gain = np.repeat(held[i // frame_size : (i + n) // frame_size], frame_size)
        if len(gain) < n:
            gain = np.pad(gain, (0, n - len(gain)), constant_values=held[-1])
        smoothed, zi = lfilter([alpha], [1.0, alpha - 1.0], gain, zi=zi)
        out[i : i + n] = audio[i : i + n] * smoothed.astype(np.float32)
    return out


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """Integrated gated loudness (ITU-R BS.1770-4), measured a block at a time.

    Same K-weighting filters and gating as pyloudnorm's Meter.integrated_loudness,
    which needs several float64 copies of the whole track.
    """
    stages = [
        (f.b, f.a, f.passband_gain, np.zeros(max(len(f.a), len(f.b)) - 1))
        for f in pyln.Meter(sr)._filters.values()
    ]
    hop = int(round(sr * 0.1))    # 400 ms gating blocks with 75 % overlap = 100 ms steps
    energies: list[np.ndarray] = []
    carry = np.empty(0)
    for i in range(0, len(audio), BLOCK):
        x = np.asarray(audio[i : i + BLOCK], dtype=np.float64)
        for k, (b, a, g, zi) in enumerate(stages):
            y, zi = lfilter(b, a, x, zi=zi)
            stages[k] = (b, a, g, zi)
            x = g * y
        x = np.concatenate([carry, x * x])
        m = len(x) // hop * hop
        energies.append(x[:m].reshape(-1, hop).sum(axis=1))
        carry = x[m:]
    e = np.concatenate(energies) if energies else np.empty(0)
    if len(e) < 4:
        return float("nan")
    z = (e[:-3] + e[1:-2] + e[2:-1] + e[3:]) / (4 * hop)
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10.0 * np.log10(z)
        above_abs = z[lj >= -70.0]
        if not above_abs.size:
            return float("-inf")
        gamma_r = -0.691 + 10.0 * np.log10(above_abs.mean()) - 10.0
        gated = z[(lj > gamma_r) & (lj > -70.0)]
        if not gated.size:
            return float("-inf")
        return float(-0.691 + 10.0 * np.log10(gated.mean()))


def normalize_track(audio: np.ndarray, sr: int, target_lufs: float, label: str = "track") -> np.ndarray:
    """Normalize a single track to target LUFS (ITU-R BS.1770), in place where possible."""
    audio = clean_audio(audio)
    if len(audio) < int(sr * 0.5):
        return audio
    if rms_level(audio) < 1e-8:
        return np.zeros_like(audio, dtype=np.float32)
    try:
        loudness = integrated_loudness(audio, sr)
    except Exception as e:
        print(f"\nWarning: loudness analysis failed ({e}); skipping {label} normalization.", file=sys.stderr)
        return audio
    if not math.isfinite(loudness):
        return audio
    gain = np.float32(10.0 ** ((target_lufs - loudness) / 20.0))
    audio = writable(audio)
    for i in range(0, len(audio), BLOCK):
        blk = audio[i : i + BLOCK]
        blk *= gain
        np.clip(blk, -1.0, 1.0, out=blk)
    return clean_audio(audio, f"normalized {label}")


def post_process(
    mic_cap: Capture | None,
    sys_cap: Capture | None,
    out_sr: int,
    denoise: bool,
    gate: bool,
    gate_threshold_db: float,
    target_lufs: float,
    out_path: Path,
) -> np.ndarray:
    mic_audio: np.ndarray | None = None
    sys_audio: np.ndarray | None = None
    mic_raw = mic_cap.load() if mic_cap else None
    sys_raw = sys_cap.load() if sys_cap else None
    has_mic, has_sys = mic_raw is not None, sys_raw is not None
    denoise, gate = denoise and has_mic, gate and has_mic

    both = has_mic and has_sys
    # Weights reflect expected relative cost; noise suppression dominates.
    W_RESAMPLE, W_DENOISE, W_GATE, W_SYS, W_LEVEL, W_NORM, W_ENCODE = 1, 10, 1, 1, 2, 2, 2
    total_weight = (
        (W_RESAMPLE if has_mic else 0)
        + (W_DENOISE if has_mic and denoise else 0)
        + (W_GATE if has_mic and gate else 0)
        + (W_SYS if has_sys else 0)
        + (W_LEVEL if both else 0)
        + W_NORM
        + W_ENCODE
    )
    print("Post-processing…", file=sys.stderr)
    pb = ProgressBar(total_weight)

    if has_mic:
        with pb.step("Mic: resample + high-pass filter", weight=W_RESAMPLE):
            mono = to_mono(mic_raw)
            del mic_raw
            resampled = resample_audio(mono, mic_cap.sample_rate, out_sr)
            del mono
            resampled = high_pass(resampled, out_sr)
        if denoise:
            with pb.step("Mic: noise suppression", weight=W_DENOISE):
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        resampled = nr.reduce_noise(
                            y=resampled,
                            sr=out_sr,
                            stationary=False,
                            prop_decrease=0.9,
                            n_jobs=1,
                            # its chunk scratch file goes beside the captures, not /tmp (tmpfs)
                            tmp_folder=str(out_path.parent),
                        ).astype(np.float32, copy=False)
                except Exception as e:
                    print(f"\nWarning: noise suppression failed ({e}); continuing without it.", file=sys.stderr)
                resampled = clean_audio(resampled, "noise-suppressed mic")
        if gate:
            with pb.step(f"Mic: noise gate ({gate_threshold_db:.0f} dB threshold)", weight=W_GATE):
                resampled = noise_gate(resampled, out_sr, threshold_db=gate_threshold_db)
        mic_audio = resampled

    if has_sys:
        with pb.step("System audio: resample", weight=W_SYS):
            mono = to_mono(sys_raw)
            del sys_raw
            sys_audio = resample_audio(mono, sys_cap.sample_rate, out_sr)
            del mono

    if mic_audio is not None and sys_audio is not None:
        with pb.step("Level matching (pre-mix LUFS)", weight=W_LEVEL):
            mic_audio = normalize_track(mic_audio, out_sr, PREMIX_LUFS, "mic track")
            sys_audio = normalize_track(sys_audio, out_sr, PREMIX_LUFS, "system track")

    mixed = mix_tracks(mic_audio, sys_audio)
    del mic_audio, sys_audio

    with pb.step("Output loudness normalization", weight=W_NORM):
        mixed = normalize_track(mixed, out_sr, target_lufs, "output")

    with pb.step("Encoding to M4A", weight=W_ENCODE):
        write_m4a(mixed, out_sr, out_path, progress_cb=pb.update_progress)

    return mixed


# --- Output encoding --------------------------------------------------------

_FFMPEG_OUT_TIME_RE = re.compile(rb"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def write_m4a(audio: np.ndarray, sample_rate: int, out_path: Path, progress_cb=None) -> None:
    """Encode float32 mono audio as AAC/M4A via ffmpeg (piped — no temp file).

    Reports real encode progress via `progress_cb(fraction)` by parsing ffmpeg's
    `-progress` stream, rather than a timer-based guess — encode time doesn't scale
    with the generic step-weight estimate, so a fake animation stalls near the cap
    while the real encode (which can be slow for long recordings) keeps running.
    """
    audio = clean_audio(audio, "final output")
    if audio.size == 0:
        sys.exit("No audio was captured.")
    total_secs = audio.size / sample_rate

    # Safety net: the fully mixed/processed audio only exists in memory at this
    # point. If the ffmpeg step below fails for any reason, sys.exit would
    # otherwise discard it with no way to recover. Stash it on disk first and
    # only remove the stash once encoding succeeds.
    backup_path = out_path.with_name(out_path.name + ".recovery.wav")
    wavfile.write(backup_path, sample_rate, audio)

    cmd = [
        "ffmpeg", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "aac", "-b:a", "192k",
        "-progress", "pipe:1", "-nostats",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def feed_stdin() -> None:
        try:
            for i in range(0, len(audio), BLOCK):
                proc.stdin.write(memoryview(np.ascontiguousarray(audio[i : i + BLOCK])).cast("B"))
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()

    stderr_chunks: list[bytes] = []

    def drain_stderr() -> None:
        stderr_chunks.append(proc.stderr.read())

    threading.Thread(target=feed_stdin, daemon=True).start()
    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    for raw_line in proc.stdout:
        m = _FFMPEG_OUT_TIME_RE.match(raw_line.strip())
        if m and progress_cb and total_secs > 0:
            h, mnt, s = m.groups()
            out_secs = int(h) * 3600 + int(mnt) * 60 + float(s)
            progress_cb(out_secs / total_secs)

    stderr_thread.join()
    returncode = proc.wait()
    if returncode != 0:
        sys.exit(
            f"ffmpeg encoding failed:\n{b''.join(stderr_chunks).decode()}\n\n"
            f"Processed audio was saved to {backup_path} — re-encode it manually, e.g.:\n"
            f"  ffmpeg -i \"{backup_path}\" -c:a aac -b:a 192k \"{out_path}\""
        )
    backup_path.unlink(missing_ok=True)


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
  uv run record.py -o ~/talks/meeting.m4a   # custom output path
  uv run record.py --upload                 # record then upload to Google Drive
  uv run record.py --list-devices           # list audio devices and exit
  uv run record.py --mic-device 2           # pick mic by device index
  uv run record.py --resume -o talk.m4a     # finish a run that died after recording
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
        "--resume", action="store_true",
        help="Don't record: post-process the capture files an earlier run left beside -o",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload output M4A to Google Drive after recording",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print audio device table and exit",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return parser


def make_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        p = Path(args.output).expanduser().resolve()
        if p.suffix.lower() != ".m4a":
            fixed = p.with_suffix(".m4a") if p.suffix else p.with_name(p.name + ".m4a")
            print(
                f"Warning: output is always encoded as AAC/M4A; using {fixed.name} instead of {p.name}",
                file=sys.stderr,
            )
            p = fixed
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

    leftover = find_captures(out_path)
    if args.resume:
        if not args.output:
            sys.exit("--resume needs -o: the output path the interrupted run was given.")
        if not leftover:
            sys.exit(f"No capture files found beside {out_path}.")
        finish(args, out_path, leftover.get("mic"), leftover.get("sys"))
        return
    if leftover:
        sys.exit(
            "Capture files from an earlier run are still beside the output:\n"
            + "".join(f"  {c.path}\n" for c in leftover.values())
            + f"Finish them with:  uv run record.py --resume -o \"{out_path}\"\n"
            "or move them away, or choose another output name."
        )

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
                            "No system audio source found. On macOS 14.2+ install the\n"
                            "Xcode Command Line Tools (xcode-select --install) so the\n"
                            "process-tap helper can be built. On older macOS install\n"
                            "BlackHole (brew install blackhole-2ch) and route output\n"
                            "through a Multi-Output Device in Audio MIDI Setup."
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
    print(f"  Raw:    {out_path.with_name(out_path.stem)}.*.raw (kept until the M4A is written)", file=sys.stderr)
    print(file=sys.stderr)

    is_win_loopback = (plat == "windows") and use_system and (
        sys_source is not None and sys_source.device_idx is not None
    )

    stop_event = threading.Event()
    mic_caps: list[Capture] = []
    sys_caps: list[Capture] = []
    mic_level: list = [0.0]
    sys_level: list = [0.0]
    threads: list[threading.Thread] = []

    if mic_device_idx is not None:
        d = sd.query_devices(mic_device_idx)
        mic_channels = min(2, max(1, int(d["max_input_channels"])))
        threads.append(threading.Thread(
            target=recording_thread_sd,
            args=(mic_device_idx, mic_channels, stop_event, mic_caps, out_path, "mic", False, mic_level),
            name="mic-recorder",
            daemon=True,
        ))

    if use_system and sys_source is not None:
        if sys_source.tap_binary:
            threads.append(threading.Thread(
                target=recording_thread_tap,
                args=(sys_source.tap_binary, stop_event, sys_caps, out_path, sys_level),
                name="sys-recorder",
                daemon=True,
            ))
        elif sys_source.parec_name:
            threads.append(threading.Thread(
                target=recording_thread_parec,
                args=(
                    sys_source.parec_name,
                    sys_source.native_sr,
                    sys_source.native_channels,
                    stop_event,
                    sys_caps,
                    out_path,
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
                    sys_caps,
                    out_path,
                    "sys",
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
                parts = [f"  {_RED}{blink} REC{_RESET}  {format_duration(elapsed)}"]
                if mic_device_idx is not None:
                    parts.append(f"  Mic: {render_vu_bar(mic_level[0])}")
                if use_system and sys_source is not None:
                    parts.append(f"  Sys: {render_vu_bar(sys_level[0])}")
                print(f"\r{fit_to_terminal(''.join(parts))}{_EL}", end="", file=sys.stderr, flush=True)
            stop_event.wait(timeout=0.1)
    except KeyboardInterrupt:
        if _IS_TTY:
            print(f"\r{_EL}", end="", file=sys.stderr)
        print("\nStopping…", file=sys.stderr)
        stop_event.set()

    for t in threads:
        t.join(timeout=5.0)

    mic_cap = mic_caps[0] if mic_caps else None
    sys_cap = sys_caps[0] if sys_caps and use_system else None
    if use_system:
        warn_if_silent(sys_cap, plat)
    finish(args, out_path, mic_cap, sys_cap)


def finish(args: argparse.Namespace, out_path: Path, mic_cap: Capture | None, sys_cap: Capture | None) -> None:
    """Post-process the capture files into out_path, then delete them.

    On any failure the capture files stay on disk and the recovery commands are
    printed. (An OOM kill can't print anything, which is why the Raw: line is
    shown when recording starts.)
    """
    caps = [c for c in (mic_cap, sys_cap) if c is not None]
    empty = [c for c in caps if c.load() is None]
    if len(empty) == len(caps):
        for c in caps:
            c.path.unlink(missing_ok=True)
        sys.exit("No audio was captured.")

    try:
        audio = post_process(
            mic_cap=mic_cap,
            sys_cap=sys_cap,
            out_sr=args.sample_rate,
            denoise=not args.no_denoise,
            gate=not args.no_gate,
            gate_threshold_db=args.gate_threshold,
            target_lufs=args.target_lufs,
            out_path=out_path,
        )
    except BaseException:
        print_recovery_help([c for c in caps if c not in empty], out_path)
        raise

    for c in caps:
        c.path.unlink(missing_ok=True)
    duration = len(audio) / args.sample_rate
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {out_path}  ({format_duration(duration)}, {size_mb:.1f} MB)", file=sys.stderr)

    if args.upload:
        upload_to_gdrive(out_path)


if __name__ == "__main__":
    main()

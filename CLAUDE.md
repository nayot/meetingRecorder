# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the script

```bash
uv run record.py                  # mic + system audio → ~/Documents/meeting_<ts>.m4a
uv run record.py --list-devices   # enumerate audio devices and exit
uv run record.py --mic-only       # microphone only
uv run record.py --system-only    # system audio only
uv run record.py --upload         # record then upload to Google Drive
```

`uv` resolves all Python dependencies declared in the PEP 723 inline script header on first run. `ffmpeg` must also be installed for the M4A encoding step.

There are no tests and no build step — the entire project is `record.py`.

## Architecture

Single-file Python CLI. Two concurrent recording threads feed `list` buffers; post-processing runs after Ctrl-C.

**Recording layer** — platform-specific:
- **Linux mic / Mac system audio / Windows WASAPI**: `sounddevice.InputStream` callback appends `float32` numpy chunks to a buffer.
- **Linux system audio**: `parec` subprocess writing to a **temp file** (not stdout). This is intentional — `parec` buffers differently for pipes vs files; piped capture produces 0 bytes even when audio is playing. After `stop_event` fires, the raw file is read into the buffer and deleted.

**Post-processing chain** (`post_process()`), mic track only:
1. High-pass Butterworth at 80 Hz (remove rumble)
2. `noisereduce` spectral suppression (`stationary=False`)
3. Frame-based noise gate with hold + smooth fade
4. Per-track LUFS normalization to −20 LUFS before mixing (equalizes mic vs system level)

System audio gets step 4 only, then both tracks are mixed, clipped, and final-normalized to `--target-lufs`.

**Output**: encoded to AAC/M4A via `ffmpeg` (piped, no temp file). Note: the README describes the output as WAV/PCM_16, but the code saves `.m4a`.

**Google Drive upload**: OAuth2 `drive.file` scope. Credentials at `~/.config/meetingRecorder/credentials.json`, token cached at `~/.config/meetingRecorder/token.json`.

## Key implementation notes

- The `parec` temp-file pattern (`recording_thread_parec`) is correct and intentional — do not switch to `subprocess.PIPE`.
- `signal.signal(SIGINT, default_int_handler)` at module load restores Ctrl-C behavior when the script is launched via `nohup ... &`.
- VU meter display is gated on `sys.stderr.isatty()` — no ANSI codes in non-TTY output.

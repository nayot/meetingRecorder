# meetingRecorder

Cross-platform single-file CLI audio recorder. Records mic + system audio
simultaneously, runs spectral noise suppression and a noise gate on the
mic, matches mic level to system audio, and saves a normalized mono WAV
to `~/Documents`. Optional one-flag upload to Google Drive.

## Install

Only [`uv`](https://docs.astral.sh/uv/) is required — Python dependencies
are declared inline (PEP 723) and resolved on first run.

```bash
git clone https://github.com/nayot/meetingRecorder.git
cd meetingRecorder/meetingRecorder
uv run record.py --list-devices    # first run downloads deps (~1 min)
```

Platform extras:

| OS      | System-audio capture                        | Extra install                                        |
| ------- | ------------------------------------------- | ---------------------------------------------------- |
| Linux   | PipeWire/PulseAudio monitor source (parec)  | `pipewire-pulse` or `pulseaudio` (provides `parec`)  |
| macOS   | Virtual loopback device                     | `brew install blackhole-2ch` + Multi-Output Device   |
| Windows | WASAPI loopback on default output           | none                                                 |

## Usage

```bash
uv run record.py                            # mic + system → ~/Documents/meeting_YYYYMMDD_HHMMSS.wav
uv run record.py --mic-only                 # mic only
uv run record.py --system-only              # system audio only
uv run record.py -o /tmp/foo.wav            # custom path
uv run record.py --upload                   # record then upload to Google Drive
uv run record.py --list-devices             # print device table and exit
```

Stop with **Ctrl-C**.

## Flags

| Flag                   | Default                           | Notes                                          |
| ---------------------- | --------------------------------- | ---------------------------------------------- |
| `--mic-only`           | —                                 | Skip system audio                              |
| `--system-only`        | —                                 | Skip mic                                       |
| `-o`, `--output PATH`  | `~/Documents/meeting_<ts>.wav`    | Custom output path                             |
| `--mic-device N`       | default input                     | Mic device index (see `--list-devices`)        |
| `--system-device N`    | platform default                  | Override system audio device index             |
| `--sample-rate Hz`     | `44100`                           | Output sample rate                             |
| `--no-denoise`         | denoise on                        | Skip spectral noise suppression on mic         |
| `--no-gate`            | gate on                           | Skip noise gate on mic                         |
| `--gate-threshold DB`  | `-65`                             | Below this RMS, mic is gated to silence        |
| `--target-lufs LUFS`   | `-16.0`                           | Final loudness target (podcast standard)       |
| `--upload`             | —                                 | Upload output to Google Drive                  |

## Mic processing chain

Applied in this order, mic only:

1. **High-pass at 80 Hz** — removes HVAC/handling rumble
2. **Spectral noise suppression** — `noisereduce`, non-stationary
3. **Noise gate** — frame-based, hold 100 ms, smooth fade 15 ms
4. **Per-track loudness match** — both tracks normalized to −20 LUFS before mixing so mic and system audio sit at the same level
5. **Final loudness normalization** — mix normalized to `--target-lufs`

System audio gets steps 4–5 only.

If quiet speech is being cut, raise the threshold:
`--gate-threshold -70`. If hiss survives gaps, lower it:
`--gate-threshold -55`.

## Google Drive upload

`--upload` uses OAuth2 with the least-privilege `drive.file` scope (can
only see files this app creates).

One-time setup:

1. https://console.cloud.google.com/apis/credentials → **Create OAuth
   client ID** → **Desktop app**
2. Enable the **Google Drive API** in the same project
3. Download the JSON and save it to `~/.config/meetingRecorder/credentials.json`

On first `--upload` run, a browser tab opens for consent; the resulting
token is cached at `~/.config/meetingRecorder/token.json`.

## Renaming & filing recordings (upload.py)

`upload.py` is a companion script for after a recording is done. For each
`.m4a` file (default: everything in `~/Documents`, or pass paths directly):

1. If the file still has `record.py`'s default name
   (`meeting_YYYYMMDD_HHMMSS.m4a`), it looks up the Google Calendar event
   whose time overlaps the recording and renames the file to
   `"<Meeting Title> - YYYY-MM-DD HHMM.m4a"`. If several events overlap,
   you're prompted to pick one. Files already given a custom name are left
   alone.
2. The (possibly renamed) file is uploaded to a shared Drive folder, then
   deleted locally — but only after the upload succeeds.

```bash
uv run upload.py                            # process every *.m4a in ~/Documents
uv run upload.py meeting_20260721_143000.m4a
uv run upload.py --dry-run                  # preview only, nothing changes
uv run upload.py --calendar-id you@example.com
```

Uses OAuth2 with the `drive` (full) and `calendar.readonly` scopes — wider
than `record.py`'s `drive.file`, since it needs to read your calendar and
write into a pre-existing shared folder rather than a file it created
itself. It reuses `~/.config/meetingRecorder/credentials.json` but caches
its own token at `~/.config/meetingRecorder/upload_token.json`; first run
opens a browser tab for the additional consent.

## Output format

Mono PCM_16 WAV at `--sample-rate` (default 44.1 kHz). This format is
universally compatible with downstream tools — feed it directly to
`ffmpeg`, Whisper, or any transcription pipeline.

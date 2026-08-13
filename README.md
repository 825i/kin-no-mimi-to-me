# 金の耳と目 — Golden Ears and Eyes

*Japanese subtitles + condensed audio for immersion learning.*

A local, offline tool for **Japanese listening/reading immersion**. Point it at a video
(or a whole library) and, for every episode, it produces two things off the **timing of an
existing subtitle track**:

1. **Japanese subtitles** — transcribed from the **spoken audio** with
   [anime-whisper](https://huggingface.co/litagin/anime-whisper), so the text matches what
   is actually *said*, not a translation of the source subtitles. → `<episode>.ja.srt`
2. **Condensed audio** — the dialogue segments cut out of the original audio and
   concatenated into one small file (silence/music/gaps removed), with metadata so it shows
   up nicely in music/audio players. → `<series>/Condensed Audio/<name>.ogg`

The key idea: an existing subtitle track (any language — usually English) already marks
*where* every line of dialogue is, with human-made, frame-accurate timing. We reuse those
timestamps as the timing oracle, then transcribe the Japanese **audio** of each segment.

> Why not just run Whisper on the whole file? Letting an ASR model find speech itself is
> where hallucination and drift creep in (e.g. "ご視聴ありがとうございました" over silent
> intros). Trusting real subtitle timings avoids that entirely — every segment we transcribe
> is guaranteed to be dialogue.

## How it works

```
video ──▶ find subtitle track ──▶ segment timings ──┬─▶ cut+concat dialogue audio ─▶ <series>/Condensed Audio/<name>.ogg
                                                     └─▶ anime-whisper per segment ─▶ <episode>.ja.srt
```

Timing source, in priority order:
1. **Embedded** subtitle track (prefers English, else the first text subtitle stream).
2. **External** subtitle file with the same basename in the same folder (`.srt/.ass/.ssa/.vtt`).

If a file has neither, it's skipped (see the optional pure-audio fallback below).

## Requirements

- **Python 3.9+**
- **FFmpeg** on your `PATH` (`ffmpeg` + `ffprobe`) — separate, non-pip install: <https://ffmpeg.org/download.html>
- Python packages from `requirements.txt` (PyTorch, Transformers, NumPy)
- A GPU is optional but much faster. The model auto-selects **CUDA** (NVIDIA) → **MPS**
  (Apple Silicon) → **CPU**.
- **For condensed audio only:** the bundled [impd](vendor/impd) needs **Bash 5+** and GNU
  tools. On macOS: `brew install bash grep findutils coreutils`. (Subtitle generation has
  no such requirement.) Condensed audio is macOS/Linux only for now; subtitles are
  cross-platform. If these are missing, the tool still makes subtitles and just skips the
  condensed audio.

## Install

```bash
# 1. clone, then create an isolated environment
python3 -m venv .venv

# 2. install Python deps
#    macOS / Linux:
.venv/bin/pip install -r requirements.txt
#    Windows (PowerShell):
#    .venv\Scripts\pip install -r requirements.txt

# 3. install FFmpeg (if you don't have it)
#    macOS:   brew install ffmpeg
#    Debian:  sudo apt install ffmpeg
#    Windows: winget install Gyan.FFmpeg   (or download from ffmpeg.org)
```

The model (~3 GB) downloads automatically from Hugging Face on first run and is cached.

### Optional: a `jpsubs` shortcut

- **macOS / Linux:** the repo ships a `subgen` launcher. Symlink it onto your `PATH`:
  ```bash
  ln -s "$(pwd)/subgen" /usr/local/bin/jpsubs   # or ~/.local/bin, /opt/homebrew/bin, …
  ```
- **Windows:** use `jpsubs.cmd` (add the repo folder to your `PATH`, or call it directly).

Everything below uses `jpsubs`; the equivalent without the shortcut is
`python subgen.py …` (or `.venv/bin/python subgen.py …`).

## Usage

```bash
# whole library (recurses into subfolders)
jpsubs --batch "/path/to/Library"

# a single file (or several)
jpsubs "/path/to/Episode.mkv"

# preview what it would do, without writing anything
jpsubs --dry-run --batch "/path/to/Library"
```

Options: `--dry-run`, `--no-subs` (condensed audio only), `--no-condensed` (subtitles only),
`--quiet`. Batch mode **resumes** — it skips outputs that already exist — and continues past
any file that fails.

### Library layout

`--batch` scans recursively and figures out the "series folder" for condensed audio,
handling both common layouts:

```
Library/TV/Cowboy Bebop/Cowboy Bebop - 01.mkv        →  Cowboy Bebop/Condensed Audio/Cowboy Bebop - 01.ogg
Library/Anime/Trigun/Episode 1/video.mkv             →  Trigun/Condensed Audio/Episode 1.ogg
```

Subtitles always land **next to the video** as `<episode filename>.ja.srt`. Condensed audio
goes in a `Condensed Audio/` folder inside the **series** directory (a folder named like an
episode — `Episode 1`, `E03`, `S01E05`, `01`, `Disc 1` — is treated as a per-episode folder,
so the series is one level up).

## Configuration

Most settings are constants at the top of `subgen.py` (audio track, preferred subtitle
language, denoise filter, condensed-audio codec/bitrate, folder name, etc.). Two things are
overridable by environment variable so you never *have* to edit the code:

| Variable | Purpose | Default |
|---|---|---|
| `JPSUBS_MODEL` | Hugging Face model id | `litagin/anime-whisper` |
| `JPSUBS_DEVICE` | Force `cuda` / `mps` / `cpu` | auto-detect |

### Updating / swapping the model

anime-whisper is referenced by its Hugging Face id, so updates are painless and can't break
the script:

- **Update to a newer revision** (if the author publishes one):
  ```bash
  .venv/bin/pip install -U "huggingface_hub[cli]"
  hf download litagin/anime-whisper   # re-fetches the latest revision into the cache
  ```
- **Try a different model** without touching the code:
  ```bash
  JPSUBS_MODEL="some-org/some-other-whisper" jpsubs --batch "/path/to/Library"
  ```

## Optional: pure-audio fallback for files with no subtitles

`generate_srt.sh` transcribes the spoken audio **without** a subtitle track (using
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) + `large-v3` and Silero VAD). It's a
separate, optional helper — it needs whisper.cpp and the ggml models installed yourself, and
is primarily tested on macOS/Linux. Use it only for content that has no subs to time against.

## Credits

This project stands on the work of others. Please respect their licenses.

- **[anime-whisper](https://huggingface.co/litagin/anime-whisper)** by **litagin** — the
  Japanese ASR model this tool runs (a Whisper fine-tune for anime/drama speech). *MIT.*
- **[kotoba-whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)** by
  **Kotoba Technologies** — anime-whisper's base model. *Apache-2.0.*
- **[Whisper](https://github.com/openai/whisper)** by **OpenAI** — the underlying speech-
  recognition architecture. *MIT.*
- **[impd](https://github.com/Ajatt-Tools/impd)** by **Ren Tatsumoto / Ajatt-Tools** —
  produces the condensed audio. impd is **vendored unmodified at [`vendor/impd`](vendor/impd)
  and remains under its own license, GPL-3.0** (not relicensed; this project merely invokes
  it as a separate program). Not affiliated with or endorsed by Ajatt-Tools.
- **[Transformers](https://github.com/huggingface/transformers)** by **Hugging Face** — runs
  the model. *Apache-2.0.*
- **[PyTorch](https://github.com/pytorch/pytorch)** — the tensor/inference backend (incl.
  CUDA and Apple MPS). *BSD-3-Clause.*
- **[FFmpeg](https://ffmpeg.org)** — audio/video decoding, cutting and encoding, called as an
  external program. *LGPL-2.1-or-later, or GPL if built with GPL components.* FFmpeg is a
  trademark of Fabrice Bellard; this project is not affiliated with or endorsed by the FFmpeg
  project.

## License

[MIT](LICENSE) © 2026 825i.

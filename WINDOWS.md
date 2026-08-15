# Running 金の耳と目 on Windows

The tool was built on macOS (Apple Silicon). Subtitle generation was already
cross-platform; condensed audio was not, because it delegates to
[impd](https://github.com/Ajatt-Tools/impd), a Bash 5 + GNU-tools program. This document
covers what changed, how to set it up, and what to do when it complains.

## Quick start

```powershell
# from the repo folder
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

That installs Python 3.12 and FFmpeg via winget if they're missing, creates `.venv`,
installs PyTorch built for your GPU, and verifies the lot. Then:

```powershell
.\jpsubs.cmd --dry-run --batch "D:\media\Anime"   # see the plan, write nothing
.\jpsubs.cmd --batch "D:\media\Anime"            # do it
.\jpsubs.cmd "D:\media\Anime\Show\Episode 01.mkv"  # one file
```

The anime-whisper model (~3 GB) downloads from Hugging Face on the first real run and is
cached in `%USERPROFILE%\.cache\huggingface`.

To use `jpsubs` from anywhere, add the repo folder to your `PATH`.

## Manual install

If you'd rather not run the script:

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
# open a NEW terminal here - winget's PATH change does not reach running shells
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# NVIDIA GPU (see the GPU note below - do NOT just `pip install torch`):
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### GPU note — install PyTorch from the CUDA index

`pip install torch` from PyPI may give you a build without kernels for your card. This
matters most on **NVIDIA 50-series (Blackwell, compute capability sm_120)**, which needs
CUDA 12.8 or newer:

| Your GPU | Wheel index |
|---|---|
| RTX 50-series (Blackwell) | `cu130` (driver 580+) or `cu128` |
| RTX 20/30/40-series | `cu128` is fine, `cu130` also works on a 580+ driver |
| No NVIDIA GPU | `cpu` — works, but transcription is many times slower |

Check what you ended up with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

Your card's `sm_XY` must appear in that arch list.

## What is different on Windows

### Condensed audio uses a native condenser, not impd

`vendor/impd` is left untouched and is still used on macOS and Linux. On Windows,
`condense.py` does the same job in Python, driving FFmpeg directly. impd cannot work on
Windows even under Git Bash / MSYS2, for three separate reasons:

1. impd writes an FFmpeg *concat list* containing POSIX paths (`/tmp/immersionpod/...`).
   The list is read by FFmpeg itself, so MSYS's argv path translation never applies, and
   native `ffmpeg.exe` resolves `/tmp/...` as `C:/tmp/...` and fails. impd then falls
   through to tagging the **uncondensed** temp audio and exits 0 — so you silently get a
   full-length 24-minute file where a condense was expected.
2. impd's `canonicalize()` only recognises paths beginning with `/`, so a Windows path is
   treated as relative and gets `$PWD` prepended: `/c/Users/you/C:\Users\you\...`.
3. MSYS re-encodes argv to the ANSI codepage when calling native binaries, so a Japanese
   filename never opens: `金の耳と目 - 02.mkv: No such file or directory`.

`condense.py` is a step-for-step port of impd v0.10's `condense`: same track-selection
weights, same 0.2 s padding, same 30 s maximum chunk, same overlap merging, same `%.6g`
timestamp formatting, same libopus flags. It was verified against impd's own `awk` on 400
randomised subtitle files plus a real episode — **402/402 chunk lists identical**, with two
deliberate exceptions, both fixes:

* impd emits a bogus `0,0` chunk when nothing survives filtering, which is what leads to
  the silent full-length fallback. We raise an error instead.
* impd's skip pattern contains `.*{\be1}.*`, but `awk` reads `\b` as a backspace, so that
  alternative can never match. We match the literal `{\be1}` ASS blur tag that was meant.

Force either backend with `JPSUBS_CONDENSER=impd` or `JPSUBS_CONDENSER=native`.

#### Why condense.py is a subprocess and not an import

`condense.py` is a port of impd, so it is a derivative work and carries impd's licence,
**GPL-3.0** — unlike the rest of the project, which is MIT. `subgen.py` therefore runs it
as a **separate program** and never imports it, which is exactly the relationship the
project already has with `vendor/impd`. Importing it would combine the two into a single
work and pull the MIT code under the GPL.

Please keep it that way. `subgen.py` has no `import condense`, and the Windows-specific
helpers it needs (`staging_path`, `path_too_long`) live in `subgen.py` because they are
original code, not derived from impd.

### `generate_srt.sh` is not ported

The optional pure-audio fallback for files with **no** subtitle track is a Bash script
around whisper.cpp. It is untouched and still macOS/Linux only. It is not needed for the
normal workflow, which times everything off an existing subtitle track.

## Environment variables

| Variable | Purpose |
|---|---|
| `JPSUBS_MODEL` | Hugging Face model id (default `litagin/anime-whisper`) |
| `JPSUBS_DEVICE` | Force `cuda` / `mps` / `cpu` |
| `JPSUBS_CONDENSER` | Force `impd` or `native` (default: auto) |
| `JPSUBS_BASH` | Path to a Bash 5, if you insist on trying impd |

## Troubleshooting

**`UnicodeEncodeError: 'charmap' codec can't encode characters`**
Fixed in the code, but if you invoke `subgen.py` directly from a shell that overrides the
encoding, set `PYTHONUTF8=1`. `jpsubs.cmd` sets it for you. This bites the moment you
redirect output to a file, because Windows then falls back to the legacy ANSI codepage
instead of UTF-8.

**`ffmpeg not found on PATH` right after installing it**
Open a new terminal. winget's PATH change does not reach shells that are already running.

**`path is 260 characters, over the 259-character Windows limit`**
Windows caps paths at 260 characters unless long paths are enabled. Either shorten the
folder names, or (as admin, then reboot):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

**Transcription runs on the CPU when you have a GPU**
`torch.cuda.is_available()` is False, or your card's `sm_XY` is missing from
`torch.cuda.get_arch_list()`. Reinstall torch from the right CUDA index (above).

**Japanese text shows as boxes in the terminal**
The text is correct; the console font lacks the glyphs. Use Windows Terminal, or check the
`.srt` file in an editor. Nothing is wrong with the output.

## Notes

* Batch mode resumes: it skips episodes whose outputs already exist, so re-running after
  an interruption is safe and cheap.
* Ctrl-C is clean. The in-progress file's partial output is discarded; finished files stay.
* Outputs are written through a short-named temporary sibling and then renamed, so an
  interrupted run never leaves a half-written `.srt` or `.ogg` in your library.

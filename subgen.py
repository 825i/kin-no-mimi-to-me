#!/usr/bin/env python3
"""
anime-jp-immersion — subtitle-timed anime-whisper + condensed audio.

For each episode it uses an existing subtitle track's TIMESTAMPS as the timing oracle
(human-made, frame-accurate) and then does two things off those same timings:

  1. CONDENSED AUDIO — cuts the dialogue segments out of the original audio and
     concatenates them into one small file (silence/music/gaps removed), the way impd
     does, with metadata so it shows up properly in music/audio players. For listening
     immersion.  ->  <Music>/<series>/Condensed Audio/<name>.ogg
     It lands in your Music folder rather than the video library so that Jellyfin/Plex
     do not index it as an episode or an extra. --condensed-dir puts it elsewhere.
  2. JAPANESE SUBTITLES — transcribes the AUDIO of each segment with anime-whisper, so
     the text matches what is actually spoken (never a translation of the source subs).
     ->  <episode filename>.ja.srt  (right next to the video)

Point it at a whole library: --batch scans recursively, handling both flat layouts
(Series/Episode.mkv) and per-episode-folder layouts (Series/Episode 1/Episode.mkv).

Timing source, in priority order:
  1. Embedded subtitle track (prefers language PREF_SUB_LANG, else the first text sub stream)
  2. External subtitle file with the same basename in the same folder (.srt/.ass/.ssa/.vtt)
If neither is found, the file is skipped.

Cross-platform: pure Python + ffmpeg. Runs on macOS, Linux and Windows. The model runs on
the best available accelerator automatically — CUDA (NVIDIA) → MPS (Apple) → CPU.

Environment overrides (so you never have to edit this file):
  JPSUBS_MODEL      Hugging Face model id to use (default: litagin/anime-whisper).
                    Point this at an updated/alternative model without touching the code.
  JPSUBS_DEVICE     Force cuda | mps | cpu (default: auto-detect).
  JPSUBS_CONDENSER  Force impd | native for condensed audio (default: auto — native on
                    Windows, the vendored impd elsewhere).
  JPSUBS_BASH       Path to a Bash 5 for impd, if it is somewhere unusual.
  JPSUBS_AUDIO_LANG Comma-separated audio language tags to prefer when a file has more
                    than one audio track (default: ja,jpn,jap,japanese).
  JPSUBS_AUDIO_TRACK Force a specific audio stream (ffmpeg's 0:a:N) instead of choosing.
  JPSUBS_COND_ROOT  Where condensed audio goes: <root>/<series>/Condensed Audio/
                    (default: the user's Music folder). Same as --condensed-dir.

Requirements: Python 3.9+, ffmpeg on PATH, and `pip install -r requirements.txt`.

Usage (via the `jpsubs` launcher, or `python subgen.py`):
    jpsubs [options] --batch <directory>
    jpsubs [options] <video> [<video> ...]
Options: --dry-run, --no-subs, --no-condensed, --quiet
"""
import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import warnings
import wave
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Windows stdio defaults to the legacy ANSI codepage (cp1252 on most installs), so the
# very first transcribed Japanese line raises UnicodeEncodeError the moment output is
# redirected to a file or a pipe. Force UTF-8 everywhere.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

# ---- Configuration (env-overridable where it matters) ----------------------
MODEL_ID = os.environ.get("JPSUBS_MODEL", "litagin/anime-whisper")


def _int_env(name):
    try:
        return int(os.environ.get(name, "").strip())
    except ValueError:
        return None


# Which audio stream to transcribe. None = choose automatically (see pick_audio_stream);
# set JPSUBS_AUDIO_TRACK=N to force ffmpeg's 0:a:N when a release is mislabelled.
AUDIO_TRACK = _int_env("JPSUBS_AUDIO_TRACK")
AUDIO_LANG = os.environ.get("JPSUBS_AUDIO_LANG", "ja,jpn,jap,japanese")
PREF_SUB_LANG = "en"                         # preferred embedded subtitle language
DENOISE_FILTER = "afftdn=nr=12"              # light denoise for the ASR audio; "" to disable
SEG_PAD_S = 0.20                             # ASR: max audio padding each side of a segment
TRAIL_GAP_SHARE = 0.25                       # of the gap to the next cue, how much the
                                             # trailing pad may take (see clamp_pads):
                                             # reaching forward is what makes a line end
                                             # with the next sentence's first word
MIN_SEG_S = 0.20                             # ignore sub segments shorter than this
NO_REPEAT_NGRAM = 5                          # anime-whisper loop guard (per model card)
SR = 16000                                   # ASR sample rate

# Condensed audio is produced by the REAL impd (github.com/Ajatt-Tools/impd), vendored at
# ./vendor/impd and invoked directly — no reimplementation. impd requires Bash 5+ and GNU
# tools (macOS: `brew install bash grep findutils coreutils`), so it is macOS/Linux only.
# On Windows we use condense.py instead, a faithful port of impd's condense algorithm
# (verified chunk-for-chunk against impd's own awk); see that file for why no Windows bash
# — Git Bash/MSYS or WSL — can drive impd correctly.
COND_DIR = "Condensed Audio"                 # the folder created under <root>/<series>/
COND_FILE_EXT = ".ogg"
# Where condensed audio is written: <COND_ROOT>/<series>/Condensed Audio/<name>.ogg.
# Empty = the user's Music folder. Deliberately OUTSIDE the video library, because
# Jellyfin and Plex will happily index a stray .ogg sitting next to the episodes and
# present it as an extra, which it is not. --condensed-dir overrides it per run.
COND_ROOT = os.environ.get("JPSUBS_COND_ROOT", "").strip()
COND_BITRATE = "32k"                         # passed to impd's config; opus VBR is plenty for speech
IMPD = Path(__file__).resolve().parent / "vendor" / "impd"
# Run as a subprocess, never imported — condense.py is GPL-3.0 (a port of impd) and this
# file is MIT. See the header of condense.py.
CONDENSE_PY = Path(__file__).resolve().parent / "condense.py"
# auto (default) | impd | native.  auto picks native on Windows, impd elsewhere when
# Bash 5 and the vendored script are both present, native otherwise.
CONDENSER = os.environ.get("JPSUBS_CONDENSER", "auto").strip().lower()

SUB_SUFFIX = ".ja.srt"
TEXT_SUB_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}
SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".m4v", ".avi", ".ts", ".webm")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_srt  # noqa: E402

VERBOSE = True


def vprint(*a):
    if VERBOSE:
        print(*a, flush=True)


# ---- Windows MAX_PATH ------------------------------------------------------
# Unless LongPathsEnabled is turned on (it is off by default), Windows caps a path at 260
# characters, 259 usable. Appending a staging suffix to an already-long output name is
# enough to cross it: a perfectly legal 251-character .ogg becomes 260 with ".part.ogg"
# and open() then fails with a bare "The system cannot find the path specified". So stage
# under a SHORT fixed-length name in the same directory — same volume, so os.replace stays
# atomic — instead of decorating the real name.
WIN_PATH_LIMIT = 259
_stage_counter = itertools.count()


def staging_path(final) -> Path:
    """A short-named sibling of `final` to write to before the atomic rename.

    Sibling (not a temp dir) so the rename stays on one volume and therefore atomic.
    Short, so it cannot push a legal path over MAX_PATH. Keeps the extension, because
    ffmpeg picks its muxer from it. Dot-prefixed, so a leftover from a crashed run is
    hidden on Unix, skipped by collect_videos, and can never be mistaken for an external
    subtitle by find_external_sub — which a name like "<episode>.ja.srt.part.srt" could.
    """
    final = Path(final)
    return final.parent / f".jpsubs-{os.getpid():x}-{next(_stage_counter):x}{final.suffix}"


def path_too_long(p) -> bool:
    return sys.platform == "win32" and len(str(Path(p).absolute())) > WIN_PATH_LIMIT


def long_path_hint(p) -> str:
    return (f"path is {len(str(Path(p).absolute()))} characters, over the "
            f"{WIN_PATH_LIMIT}-character Windows limit:\n      {p}\n"
            "    Either shorten the folder names, or enable long paths (admin, then "
            "reboot):\n"
            '      New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\'
            'FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force')


def tempdir(prefix):
    """A TemporaryDirectory that tolerates Windows briefly holding a handle open on a
    file an ffmpeg child just wrote (antivirus/indexer), which would otherwise turn a
    successful run into a PermissionError during cleanup."""
    if sys.version_info >= (3, 10):
        return tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)
    return tempfile.TemporaryDirectory(prefix=prefix)


# ---- Small helpers ---------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def run(cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def sub_output_for(video: Path) -> Path:
    # Subtitles stay right next to the video: <episode filename>.ja.srt
    return Path(str(video.with_suffix("")) + SUB_SUFFIX)


def _looks_like_episode_folder(name: str) -> bool:
    """True if a folder name looks like a per-episode label (e.g. 'Episode 1', 'E03',
    'S01E05', '01', 'Disc 1') rather than a series title. Deliberately strict so real
    series titles (even 'Episode of Bardock') aren't mistaken for episode folders."""
    n = name.strip()
    if re.fullmatch(r"\d{1,4}", n):
        return True
    if re.fullmatch(r"(?i)(ep|episode|e|disc|vol|part)\s*\.?\s*\d{1,4}", n):
        return True
    if re.search(r"(?i)\bS\d{1,2}E\d{1,3}\b", n):
        return True
    return False


def series_dir_for(video: Path, root: Path) -> Path:
    """Best-effort 'series directory' for a video within a library tree.
    The video's parent is the series dir, UNLESS that parent is a per-episode folder
    (name looks like an episode label AND it holds at most one video), e.g.
    '.../Series/Episode 1/ep.mkv', in which case the series is one level up.
    Never climbs above `root`."""
    parent = video.parent
    if parent == root or not _looks_like_episode_folder(parent.name):
        return parent
    try:
        n_vids = sum(1 for p in parent.iterdir()
                     if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    except OSError:
        n_vids = 99
    if n_vids <= 1:
        grand = parent.parent
        if grand == root or root in grand.parents:
            return grand
    return parent


def _win_music_folder():
    """Windows' real Music folder. Not %USERPROFILE%\\Music — OneDrive redirects it, so
    ask the shell for the known folder instead of guessing."""
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        # FOLDERID_Music {4BD8D571-6D19-48D3-BE97-422220080E43}
        fid = GUID(0x4BD8D571, 0x6D19, 0x48D3,
                   (ctypes.c_ubyte * 8)(0xBE, 0x97, 0x42, 0x22, 0x20, 0x08, 0x0E, 0x43))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(fid), 0, None, ctypes.byref(out)) == 0:
            try:
                return Path(out.value) if out.value else None
            finally:
                ctypes.windll.ole32.CoTaskMemFree(out)
    except Exception:
        pass
    return None


def music_dir() -> Path:
    """The user's Music folder, honouring Windows known-folder redirection and the XDG
    user-dirs config on Linux. Falls back to ~/Music."""
    if sys.platform == "win32":
        p = _win_music_folder()
        if p:
            return p
    elif sys.platform.startswith("linux"):
        cfg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        try:
            for line in (cfg / "user-dirs.dirs").read_text(
                    encoding="utf-8", errors="replace").splitlines():
                m = re.match(r'\s*XDG_MUSIC_DIR\s*=\s*"(.*)"\s*$', line)
                if m:
                    return Path(m.group(1).replace("$HOME", str(Path.home())))
        except OSError:
            pass
    return Path.home() / "Music"


def cond_root() -> Path:
    return Path(COND_ROOT).expanduser() if COND_ROOT else music_dir()


def _looks_like_season_folder(name: str) -> bool:
    """'Season 1', 'S02', 'Series 3' — a subdivision of a show, not the show itself."""
    return bool(re.fullmatch(r"(?i)(season|series|s)\s*\.?\s*\d{1,2}", name.strip()))


def series_name_for(video: Path, root: Path) -> str:
    """The show's name, for grouping output. Collapses 'Season N' folders onto their
    parent so every season of a show lands in one place instead of a shared 'Season 1'."""
    sdir = series_dir_for(video, root)
    if _looks_like_season_folder(sdir.name) and sdir.parent.name:
        return sdir.parent.name
    return sdir.name


def cond_output_for(video: Path, root: Path) -> Path:
    """Condensed audio path: <Music or COND_ROOT>/<series>/<COND_DIR>/<name>.ogg.

    Written outside the video library on purpose (see COND_ROOT). Uses the episode-folder
    name for per-episode layouts (more meaningful, and avoids collisions when the inner
    files are generically named), else the video's own name."""
    sdir = series_dir_for(video, root)
    base = video.with_suffix("").name if video.parent == sdir else video.parent.name
    return cond_root() / series_name_for(video, root) / COND_DIR / (base + COND_FILE_EXT)


# ---- Subtitle discovery ----------------------------------------------------
def list_subtitle_streams(video: Path):
    cp = run(["ffprobe", "-v", "error", "-select_streams", "s",
              "-show_entries", "stream=index,codec_name:stream_tags:stream_disposition=default,forced",
              "-of", "json", str(video)])
    if cp.returncode != 0:
        return []
    try:
        streams = json.loads(cp.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []
    out = []
    for rel, s in enumerate(streams):
        tags = s.get("tags", {}) or {}
        disp = s.get("disposition", {}) or {}
        nframes = 0
        for k, v in tags.items():  # mkv statistics tag (may be lang-suffixed) = event count
            if k.upper().startswith("NUMBER_OF_FRAMES"):
                try:
                    nframes = max(nframes, int(v))
                except (TypeError, ValueError):
                    pass
        out.append({
            "rel": rel, "codec": s.get("codec_name", ""),
            "lang": (tags.get("language") or "").lower(), "title": tags.get("title") or "",
            "nframes": nframes, "default": disp.get("default", 0), "forced": disp.get("forced", 0),
        })
    return out


_SIGNS_SONGS_RE = re.compile(r"(?i)sign|song|karaoke|forced|comment|lyric")
_FULL_RE = re.compile(r"(?i)full|dialog")


def pick_embedded_stream(streams):
    """Pick the FULL DIALOGUE subtitle track. Fansubs frequently ship a separate
    'Signs/Songs' track (styling only, no dialogue) — picking that transcribes the OP/ED
    song instead of dialogue. Prefer full/dialogue titles in the preferred language with
    the most events; strongly avoid signs/songs/forced tracks."""
    text = [s for s in streams if s["codec"] in TEXT_SUB_CODECS]
    if not text:
        return None

    def score(s):
        sc = 0.0
        if s["lang"].startswith(PREF_SUB_LANG):
            sc += 1000
        if _SIGNS_SONGS_RE.search(s["title"]):
            sc -= 5000
        if _FULL_RE.search(s["title"]):
            sc += 500
        if s["forced"]:
            sc -= 2000
        if s["default"]:
            sc += 100
        sc += min(s["nframes"], 5000) / 10.0  # more events -> more likely the dialogue track
        return sc

    return max(text, key=score)["rel"]


def find_external_sub(video: Path):
    stem = video.with_suffix("").name
    candidates = []
    for p in video.parent.iterdir():
        if not p.is_file() or p.suffix.lower() not in SUB_EXTS:
            continue
        if p.name.endswith(SUB_SUFFIX):
            continue  # never treat our own output as a timing source
        if p.name == stem + p.suffix or p.name.startswith(stem + "."):
            candidates.append(p)
    if not candidates:
        return None
    for p in candidates:
        if f".{PREF_SUB_LANG}" in p.name.lower():
            return p
    return sorted(candidates)[0]


# Style-name tokens that mark a subtitle line as NOT spoken dialogue (OP/ED songs,
# karaoke, typeset signs, titles/credits). Songs make anime-whisper loop on sung audio.
_SKIP_STYLE_TOKENS = {
    "op", "ed", "opening", "ending", "song", "songs", "karaoke", "kara", "insert",
    "lyric", "lyrics", "sign", "signs", "title", "titles", "credit", "credits",
    "logo", "caption", "romaji",
}


def _is_song_or_sign_style(style: str) -> bool:
    return any(t in _SKIP_STYLE_TOKENS for t in re.split(r"[^a-z0-9]+", style.lower()) if t)


def _ass_time(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _strip_ass_tags(text: str) -> str:
    text = re.sub(r"\{[^}]*\}", "", text)      # {\...} override blocks
    text = re.sub(r"\\[Nnh]", " ", text)       # \N \n \h line breaks/spaces
    return re.sub(r"\s+", " ", text).strip()


def filter_ass(ass_text: str):
    """Parse ASS/SSA Dialogue events, keeping SPOKEN DIALOGUE only — dropping song/sign/
    karaoke lines (by style name, karaoke \\k tags, or vector drawings). Returns
    [(start, end, text)] with override tags stripped."""
    entries, fmt, in_events = [], None, False
    for ln in ass_text.splitlines():
        low = ln.strip().lower()
        if low.startswith("[") and low.endswith("]"):
            in_events = (low == "[events]")
            continue
        if in_events and low.startswith("format:"):
            fmt = [x.strip().lower() for x in ln.split(":", 1)[1].split(",")]
        elif in_events and low.startswith("dialogue:") and fmt:
            vals = [v.strip() for v in ln.split(":", 1)[1].split(",", len(fmt) - 1)]
            row = dict(zip(fmt, vals))
            style, text = row.get("style", ""), row.get("text", "")
            if _is_song_or_sign_style(style):
                continue
            if "\\k" in text or re.search(r"\\p[1-9]", text):   # karaoke / vector sign
                continue
            clean = _strip_ass_tags(text)
            if not any(unicodedata.category(c)[0] in ("L", "N") for c in clean):
                continue
            try:
                st, en = _ass_time(row["start"]), _ass_time(row["end"])
            except Exception:
                continue
            if en - st >= MIN_SEG_S:
                entries.append((st, en, clean))
    entries.sort()
    return entries


def _write_srt_tuples(entries, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(entries, 1):
            f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{t}\n\n")


def obtain_timing_srt(video: Path, work_srt: Path):
    streams = list_subtitle_streams(video)
    rel = pick_embedded_stream(streams)
    if rel is not None:
        codec = next((s["codec"] for s in streams if s["rel"] == rel), "")
        if codec in ("ass", "ssa"):
            raw = work_srt.with_name("raw.ass")
            cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(video), "-map", f"0:s:{rel}", str(raw)])
            if cp.returncode == 0 and raw.exists():
                ents = filter_ass(raw.read_text(encoding="utf-8", errors="replace"))
                if ents:
                    _write_srt_tuples(ents, work_srt)
                    return f"embedded ASS 0:s:{rel} — dialogue only ({len(ents)} lines)"
        else:
            cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(video), "-map", f"0:s:{rel}", str(work_srt)])
            if cp.returncode == 0 and work_srt.exists() and work_srt.stat().st_size > 0:
                return f"embedded sub stream 0:s:{rel}"
    if streams and rel is None:
        codecs = ", ".join(sorted({s["codec"] for s in streams}))
        vprint(f"    (embedded subs present but not text-based: {codecs}; can't use for timing)")
    ext = find_external_sub(video)
    if ext is not None:
        if ext.suffix.lower() in (".ass", ".ssa"):
            ents = filter_ass(ext.read_text(encoding="utf-8", errors="replace"))
            if ents:
                _write_srt_tuples(ents, work_srt)
                return f"external ASS '{ext.name}' — dialogue only ({len(ents)} lines)"
        else:
            cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                      "-i", str(ext), str(work_srt)])
            if cp.returncode == 0 and work_srt.exists() and work_srt.stat().st_size > 0:
                return f"external sub file '{ext.name}'"
    return None


def parse_timings(srt_path: Path):
    try:
        text = srt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = srt_path.read_text(encoding="utf-8", errors="replace")
    spans = [(e["start"], e["end"]) for e in clean_srt.parse_srt(text)
             if e["end"] - e["start"] >= MIN_SEG_S]
    spans.sort()
    return spans


# ---- Condensed audio (delegated to the real impd) --------------------------
def parse_episode_number(name: str):
    for pat in (r'[Ss]\d{1,2}[Ee](\d{1,3})', r'[Ee]p?\.?\s*(\d{1,3})\b', r'[-#]\s*(\d{1,3})\b'):
        m = re.search(pat, name)
        if m:
            return str(int(m.group(1)))
    return None


def gather_metadata(video: Path, root: Path):
    cp = run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(video)])
    vtags = {}
    try:
        vtags = (json.loads(cp.stdout).get("format", {}) or {}).get("tags", {}) or {}
    except json.JSONDecodeError:
        pass
    sdir = series_dir_for(video, root)
    series = series_name_for(video, root)   # matches the output folder; 'Season 1' collapsed
    # episode label: the video name for flat layouts, the episode-folder name for nested
    base = video.with_suffix("").name if video.parent == sdir else video.parent.name
    meta = {
        "title": vtags.get("title") or base,
        "album": series,               # series -> groups episodes together in players
        "artist": vtags.get("artist") or series,
        "album_artist": series,
        "genre": "Japanese",
        "comment": "Condensed dialogue audio for immersion (generated from subtitle timings).",
    }
    ep = parse_episode_number(base) or parse_episode_number(video.with_suffix("").name)
    if ep:
        meta["track"] = ep
    if vtags.get("date"):
        meta["date"] = vtags["date"]
    return meta


_BASH5 = "?"


def find_bash5():
    """Locate a Bash >= 5 (impd requires it). macOS system bash is 3.2; Homebrew provides 5.

    Returns None on Windows unless JPSUBS_BASH forces a specific interpreter. Neither
    Windows bash can actually drive impd: Git Bash/MSYS re-encodes argv to the ANSI
    codepage (so Japanese filenames fail to open) and native ffmpeg cannot resolve the
    POSIX paths impd writes into its concat list; WSL's bash sees a different filesystem
    altogether. condense.py handles Windows instead.
    """
    global _BASH5
    if _BASH5 != "?":
        return _BASH5
    _BASH5 = None
    explicit = os.environ.get("JPSUBS_BASH", "").strip()
    if not explicit and sys.platform == "win32":
        return None
    candidates = ([explicit] if explicit else
                  ["/opt/homebrew/bin/bash", "/usr/local/bin/bash",
                   shutil.which("bash"), "/bin/bash"])
    seen = []
    for c in candidates:
        if not c or c in seen or not os.path.exists(c):
            continue
        seen.append(c)
        try:
            v = subprocess.run([c, "-c", "echo ${BASH_VERSINFO[0]:-0}"], capture_output=True, text=True)
            if v.stdout.strip().isdigit() and int(v.stdout.strip()) >= 5:
                _BASH5 = c
                break
        except Exception:
            pass
    return _BASH5


def which_condenser() -> str:
    """Which condensed-audio backend to use: 'impd' or 'native'."""
    if CONDENSER in ("impd", "native"):
        return CONDENSER
    if sys.platform == "win32":
        return "native"
    return "impd" if (find_bash5() and IMPD.exists()) else "native"


def _tag_and_place(raw: Path, out_path: Path, meta: dict):
    """Copy `raw` to `out_path` adding player metadata, by stream copy (no re-encode),
    written atomically. Both condensers strip tags, so this is where they get set."""
    tmp_out = staging_path(out_path)
    tag = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
           "-c", "copy", "-map_metadata", "-1"]
    for k, v in meta.items():
        if v:
            tag += ["-metadata", f"{k}={v}"]
    tag += [str(tmp_out)]
    try:
        cp = run(tag)
    except BaseException:
        tmp_out.unlink(missing_ok=True)
        raise
    if cp.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"tagging failed: {cp.stderr.strip()[:200]}")
    os.replace(tmp_out, out_path)


def make_condensed(video: Path, timing_srt: Path, out_path: Path, meta: dict):
    """Make the condensed audio, then add player metadata.

    Dispatches to the vendored impd on macOS/Linux and to the native Python port on
    Windows. Both cut each dialogue line independently, so the result is clean — no
    boundary stutter."""
    backend = which_condenser()
    if backend == "impd" and not (find_bash5() and IMPD.exists()):
        if sys.platform == "win32":
            vprint("    JPSUBS_CONDENSER=impd was requested, but impd cannot run on "
                   "Windows — using the native condenser instead.")
        else:
            vprint("    impd unavailable (needs Bash 5 + GNU tools; macOS: "
                   "brew install bash grep findutils coreutils) — using the native "
                   "condenser instead.")
        backend = "native"
    if backend == "native":
        return make_condensed_native(video, out_path, meta)
    return make_condensed_impd(video, out_path, meta)


def make_condensed_native(video: Path, out_path: Path, meta: dict):
    """condense.py — impd's algorithm in pure Python + ffmpeg. Never writes full-length
    audio in place of a condense; it fails instead.

    condense.py is run as a SEPARATE PROGRAM, deliberately, and is never imported. It is a
    port of impd and therefore GPL-3.0, while this file is MIT — invoking it at arm's
    length is the same relationship this project already has with vendor/impd. Do not
    replace this with `import condense`.
    """
    if not CONDENSE_PY.exists():
        raise RuntimeError(f"condense.py is missing from {CONDENSE_PY.parent}")
    with tempdir("subgen.cond.") as td:
        tdp = Path(td)
        raw, info_json = tdp / "cond.ogg", tdp / "info.json"
        cmd = [sys.executable, str(CONDENSE_PY), str(video), "-o", str(raw),
               "--bitrate", COND_BITRATE, "--json-out", str(info_json)]
        if not VERBOSE:
            cmd.append("--quiet")
        # Stream its output line by line so per-chunk progress still appears live.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env=dict(os.environ, PYTHONUTF8="1"))
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            del tail[:-6]
            vprint(line)
        if proc.wait() != 0:
            raise RuntimeError("condensing failed: " + " / ".join(tail)[-300:])
        try:
            info = json.loads(info_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {}
        _tag_and_place(raw, out_path, meta)
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    extra = ""
    if info.get("dialogue_s") is not None and info.get("source_s"):
        extra = (f", {info['dialogue_s'] / 60:.1f} min of dialogue from "
                 f"{info['source_s'] / 60:.1f} min")
    vprint(f"    -> {out_path.name}  ({size_mb:.1f} MB{extra})")


def make_condensed_impd(video: Path, out_path: Path, meta: dict):
    """Make the condensed audio with the REAL impd (vendored at ./vendor/impd), then add
    player metadata (impd strips tags). Requires Bash 5+ and GNU tools."""
    bash5 = find_bash5()
    with tempdir("subgen.impd.") as td:
        tdp = Path(td)
        # Isolate impd fully: give it a throwaway config + library dir so it never touches
        # ~/Music, ~/.config, etc. This also lets us set the opus bitrate.
        (tdp / "impd").mkdir()
        (tdp / "impd" / "config").write_text(f"music_dir={tdp}\nbitrate={COND_BITRATE}\n", encoding="utf-8")
        env = dict(os.environ, XDG_CONFIG_HOME=str(tdp))
        raw = tdp / "cond.ogg"
        # Let impd auto-detect BOTH the subtitle and audio tracks — its own proven logic.
        # We deliberately do NOT pass -s (our filtered subs): coupling condensing to our
        # subtitle filter produced broken/near-empty condenses. impd on its own is flawless.
        # (No -a either: impd's -a is an absolute stream index; passing 0 maps the video.)
        cmd = [bash5, str(IMPD), "condense", "-i", str(video), "-o", str(raw)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        # Reformat impd's own "Processing chunk from X to Y" lines into our per-line style.
        if VERBOSE:
            for line in (proc.stdout or "").splitlines():
                m = re.search(r"[Cc]hunk from (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)", line)
                if m:
                    s0, e0 = float(m.group(1)), float(m.group(2))
                    vprint(f"      cut [{mmss(s0)} → {mmss(e0)}]  {e0 - s0:4.1f}s")
        if proc.returncode != 0 or not raw.exists() or raw.stat().st_size == 0:
            raise RuntimeError("impd condense failed: " + ((proc.stdout or "") + (proc.stderr or ""))[-300:])
        _tag_and_place(raw, out_path, meta)
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    vprint(f"    -> {out_path.name}  ({size_mb:.1f} MB)")


# ---- ASR (anime-whisper) ---------------------------------------------------
_AUDIO_LANGS = {x.strip().lower() for x in AUDIO_LANG.split(",") if x.strip()}
# Titles used for director's commentary and audio-description tracks. Never transcribe one.
_COMMENTARY_RE = re.compile(r"(?i)comment|解説|audio\s*description|descriptive")


def list_audio_streams(video: Path):
    """Relative index, language, title and 'is this a commentary track' per audio stream."""
    cp = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=index,codec_name,channels:stream_tags=language,title:"
              "stream_disposition=comment,visual_impaired", "-of", "json", str(video)])
    if cp.returncode != 0:
        return []
    try:
        streams = json.loads(cp.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []
    out = []
    for rel, s in enumerate(streams):
        tags = s.get("tags", {}) or {}
        disp = s.get("disposition", {}) or {}
        title = tags.get("title") or ""
        out.append({
            "rel": rel,
            "lang": (tags.get("language") or "").lower(),
            "title": title,
            "commentary": bool(disp.get("comment") or disp.get("visual_impaired")
                               or _COMMENTARY_RE.search(title)),
        })
    return out


def pick_audio_stream(video: Path):
    """(relative index for ffmpeg's 0:a:N, reason) — the track we should transcribe.

    Dual-audio releases routinely list the English dub FIRST, so taking 0:a:0 blindly
    transcribes the dub — which anime-whisper faithfully renders as katakana-ised English.
    Prefer a Japanese-tagged stream; never pick a commentary or audio-description track;
    fall back to the first stream when a release carries no language tags at all (which is
    exactly the old behaviour, so single-track files are unaffected).

    Deliberately simple: pick by language tag, not by a weighting table. Releases in the
    wild break every other signal — a Kaijuu 8-gou encode marks BOTH its Japanese and
    English tracks `default`, and plenty of rips carry no tags whatsoever.
    """
    if AUDIO_TRACK is not None:
        return AUDIO_TRACK, "forced by JPSUBS_AUDIO_TRACK"
    streams = list_audio_streams(video)
    if not streams:
        return 0, "no audio stream info"
    usable = [s for s in streams if not s["commentary"]] or streams
    preferred = [s for s in usable if s["lang"] in _AUDIO_LANGS]
    if preferred:
        why = preferred[0]["lang"]
        if len(preferred) > 1:
            why += f", first of {len(preferred)}"
        if len(streams) > len(usable):
            why += f", skipped {len(streams) - len(usable)} commentary"
        return preferred[0]["rel"], why
    if any(s["lang"] for s in streams):
        tags = ", ".join(s["lang"] or "untagged" for s in streams)
        return usable[0]["rel"], f"NO JAPANESE TRACK — only [{tags}]"
    return usable[0]["rel"], "no language tags"


def extract_asr_audio(video: Path, wav: Path, track=None):
    if track is None:
        track = pick_audio_stream(video)[0]
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-map", f"0:a:{track}"]
    if DENOISE_FILTER:
        args += ["-af", DENOISE_FILTER]
    args += ["-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(wav)]
    cp = run(args)
    if cp.returncode != 0:
        raise RuntimeError(f"audio extraction failed: {cp.stderr.strip()[:300]}")


def load_wav_f32(path: Path):
    """Read a 16-bit mono PCM WAV into a float32 numpy array in [-1, 1]. No heavy deps."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0


_PIPE = None


def _select_device():
    import torch
    forced = os.environ.get("JPSUBS_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        device = forced
    elif torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    return device, dtype


def model_is_available(model_id: str) -> bool:
    """True if the model is already local (a path, or cached from a previous run), so we
    only mention downloading when a download will actually happen."""
    if os.path.isdir(model_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    except Exception:
        return False


def get_pipe():
    global _PIPE
    if _PIPE is None:
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as e:
            sys.exit("Error: missing Python dependencies. Activate your venv and run:\n"
                     "    pip install -r requirements.txt\n"
                     f"(import error: {e})")
        device, dtype = _select_device()
        if model_is_available(MODEL_ID):
            vprint(f"    Loading {MODEL_ID} on {device}...")
        else:
            vprint(f"    Downloading {MODEL_ID} (one-time, ~3 GB), then loading on {device}...")
        kw = dict(model=MODEL_ID, device=device, chunk_length_s=30.0, batch_size=1)
        try:
            _PIPE = pipeline("automatic-speech-recognition", dtype=dtype, **kw)
        except TypeError:
            # transformers < 4.56 spells this torch_dtype; it became `dtype` later.
            _PIPE = pipeline("automatic-speech-recognition", torch_dtype=dtype, **kw)
    return _PIPE


def clamp_pads(spans, pad=SEG_PAD_S):
    """Return [(start, end, lead_pad, trail_pad)] with the ASR padding clamped so that a
    window never reaches into a neighbouring cue's speech.

    Real subtitle releases often separate consecutive cues by only ~0.08 s. With a flat
    0.2 s pad on each side, every one of those windows ran ~0.12 s past the next cue's
    start, so anime-whisper heard — and duly transcribed — the first mora or two of the
    next sentence onto the end of this line. The next line then began from its own second
    word, because its window started after that audio had already gone by. Both halves
    read as though a word had been shifted between them, e.g.

        甘やかされたから。おや      <- 'おや' is the 親 that opens the NEXT line
        親にも世間にもな。

    The two sides are NOT treated alike, because they fail differently:

      * Reaching FORWARD is what causes the bleed above, so the trailing pad only takes a
        small share of the gap (TRAIL_GAP_SHARE). It still needs a little room, or a cue
        that ends a hair early clips the last mora.
      * Reaching BACKWARD is how the model hears a word's onset — clip that and the first
        word degrades (もともと was heard as そもと when the lead pad was cut to 0.04 s).
        The previous cue's speech has already ended by its own end time, so the lead pad
        may safely take the WHOLE gap: it gains onset room without hearing the previous
        line's words.

    Consecutive windows may therefore share a sliver of the silence in the gap, which
    costs nothing, but neither one reaches into the other's speech.
    """
    out = []
    for i, (s, e) in enumerate(spans):
        prev_end = spans[i - 1][1] if i else None
        next_start = spans[i + 1][0] if i + 1 < len(spans) else None
        lead = pad if prev_end is None else min(pad, max(0.0, s - prev_end))
        trail = (pad if next_start is None else
                 min(pad, max(0.0, (next_start - e) * TRAIL_GAP_SHARE)))
        out.append((s, e, lead, trail))
    return out


def transcribe_span(pipe, audio, start, end, lead=SEG_PAD_S, trail=SEG_PAD_S):
    a = max(0, int((start - lead) * SR))
    b = min(len(audio), int((end + trail) * SR))
    if b <= a:
        return ""
    out = pipe({"raw": audio[a:b], "sampling_rate": SR},
               generate_kwargs={"language": "Japanese", "no_repeat_ngram_size": NO_REPEAT_NGRAM})
    return (out.get("text") or "").strip()


def _has_content(t: str) -> bool:
    return any(unicodedata.category(c)[0] in ("L", "N") for c in t)


def dedup_spans(spans):
    """Merge heavily-overlapping segments (>50% of the shorter one) into a single window.
    Karaoke/song layers and duplicate events share (near-)identical timings; without this
    the same audio is transcribed several times and the lines stack on screen. Sequential
    dialogue with only small readability overlaps stays separate."""
    if not spans:
        return []
    out = [list(spans[0])]
    for s, e in sorted(spans):
        ps, pe = out[-1]
        ov = min(pe, e) - max(ps, s)
        shorter = min(pe - ps, e - s)
        if shorter > 0 and ov / shorter > 0.5:
            out[-1][1] = max(pe, e)     # same window -> transcribe once
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def enforce_non_overlap(entries, min_gap=0.04):
    """Trim each subtitle so it ends before the next begins — players stack overlaps."""
    for i in range(len(entries) - 1):
        nxt = entries[i + 1]["start"]
        if entries[i]["end"] > nxt - min_gap:
            entries[i]["end"] = max(entries[i]["start"] + 0.1, nxt - min_gap)
    return entries


def _is_loop(t: str) -> bool:
    """Detect an anime-whisper repetition loop (it loops on sung/music audio): a short unit
    repeated 3+ times — back-to-back or with punctuation between — that dominates the line
    (e.g. '君のせい君のせい君のせい…' or '君のせい君のせい、君のせいで、私…')."""
    m = re.search(r"(.{2,15}?)(?:[\s、。,.・…!?！？ー~]*\1){2,}", t)
    if m and len(m.group(0)) >= max(8, 0.55 * len(t)):
        return True
    for L in range(3, 8):  # a short phrase occurring 3+ times, covering >= half the line
        for i in range(min(len(t) - L + 1, 12)):
            u = t[i:i + L]
            if u.strip() and t.count(u) >= 3 and t.count(u) * L >= 0.5 * len(t):
                return True
    return False


def cleanup(entries):
    out = []
    for e in entries:
        t = e["text"].strip()
        if not _has_content(t):
            continue
        if any(m in t for m in clean_srt.HALLUCINATION_MARKERS):
            continue
        if _is_loop(t):  # runaway repetition (usually sung/music audio)
            continue
        if out and t == out[-1]["text"]:  # drop consecutive duplicate lines
            continue
        out.append({"start": e["start"], "end": e["end"], "text": t})
    return out


def write_srt(entries, path: Path):
    tmp = staging_path(path)  # atomic: write then rename
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for i, e in enumerate(entries, 1):
                f.write(f"{i}\n{fmt_ts(e['start'])} --> {fmt_ts(e['end'])}\n{e['text']}\n\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_subtitles(video: Path, spans, out_path: Path):
    with tempdir("subgen.asr.") as td:
        wav = Path(td) / "audio.wav"
        track, why = pick_audio_stream(video)
        note = f"denoise {DENOISE_FILTER}" if DENOISE_FILTER else "no denoise"
        vprint(f"    Extracting ASR audio from 0:a:{track} ({why}); 16kHz mono, {note}...")
        if "NO JAPANESE" in why:
            # Worth saying even under --quiet: the transcript will be of the wrong language.
            print(f"    WARNING: {video.name} has no Japanese audio track; "
                  f"transcribing 0:a:{track} instead.")
        extract_asr_audio(video, wav, track)
        audio = load_wav_f32(wav)
        spans = dedup_spans(spans)  # collapse overlapping karaoke/duplicate windows
        windows = clamp_pads(spans)  # keep each window out of its neighbours' speech
        pipe = get_pipe()
        vprint("")  # separate the model-loading progress bar from the transcription
        vprint(f"    Transcribing {len(spans)} segments with anime-whisper:")
        entries = []
        for start, end, lead, trail in windows:
            text = transcribe_span(pipe, audio, start, end, lead, trail)
            entries.append({"start": start, "end": end, "text": text})
            if text:
                vprint(f"      [{mmss(start)}] {text}")
    entries = enforce_non_overlap(cleanup(entries))
    write_srt(entries, out_path)
    vprint(f"    -> {out_path.name}  ({len(entries)} lines)")


# ---- Per-file driver -------------------------------------------------------
def process_one(video: Path, root: Path, skip_existing: bool, dry_run: bool,
                want_subs: bool, want_cond: bool) -> str:
    if not video.is_file():
        print(f"Error: {video} not found — skipping\n")
        return "missing"

    sub_out = sub_output_for(video)
    cond_out = cond_output_for(video, root)
    do_subs = want_subs and not (skip_existing and sub_out.exists())
    do_cond = want_cond and not (skip_existing and cond_out.exists())
    if not do_subs and not do_cond:
        print(f"=== Skipping (outputs exist): {video.name}\n")
        return "skipped"

    print(f"=== Processing: {video.relative_to(root) if root in video.parents else video.name}")
    plan = (["condensed audio"] if do_cond else []) + (["subtitles"] if do_subs else [])
    vprint(f"    Will produce: {', '.join(plan)}")

    # Fail early and legibly rather than deep inside ffmpeg with a bare "cannot find the
    # path specified".
    for label, target in (("subtitles", sub_out if do_subs else None),
                          ("condensed audio", cond_out if do_cond else None)):
        if target is not None and path_too_long(target):
            print(f"    ERROR: cannot write {label} — {long_path_hint(target)}\n")
            return "failed"

    try:
        with tempdir("subgen.") as td:
            work_srt = Path(td) / "timing.srt"
            source = obtain_timing_srt(video, work_srt)
            if source is None:
                print("    No usable subtitle track (embedded or external) — skipping.\n")
                return "nosubs"
            spans = parse_timings(work_srt)
            vprint(f"    Timing from {source}: {len(spans)} segments")
            if not spans:
                print("    Subtitle file had no usable timings — skipping.\n")
                return "nosubs"

            if dry_run:
                if do_cond:
                    vprint(f"    [dry-run] condensed audio -> {cond_out}")
                if do_subs:
                    track, why = pick_audio_stream(video)
                    vprint(f"    [dry-run] ASR audio      -> 0:a:{track} ({why})")
                    if "NO JAPANESE" in why:
                        print(f"    WARNING: no Japanese audio track in {video.name}")
                    vprint(f"    [dry-run] subtitles      -> {sub_out}")
                print(f"    [dry-run] from {len(spans)} segments\n")
                return "ok"

            if do_cond:
                vprint("")
                vprint(f"    Condensing with {which_condenser()}...")
                cond_out.parent.mkdir(parents=True, exist_ok=True)
                make_condensed(video, work_srt, cond_out, gather_metadata(video, root))
            if do_subs:
                vprint("")
                make_subtitles(video, spans, sub_out)
    except KeyboardInterrupt:
        vprint("    Interrupted — in-progress output for this file discarded.")
        raise
    except Exception as e:
        print(f"    ERROR: {e!r}\n")
        return "failed"

    print("    Done.\n")
    return "ok"


def collect_videos(directory: Path):
    """Recursively find video files under `directory`, skipping hidden folders and the
    condensed-audio output folders."""
    vids = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != COND_DIR]
        for fn in filenames:
            if fn.startswith("."):
                continue  # skip dotfiles, incl. macOS ._AppleDouble sidecars on SMB shares
            p = Path(dirpath) / fn
            if p.suffix.lower() in VIDEO_EXTS:
                vids.append(p)
    return sorted(vids)


def check_tools():
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        hint = "Install ffmpeg — see https://ffmpeg.org/download.html"
        if sys.platform == "win32":
            hint = ("Install it with:  winget install Gyan.FFmpeg\n"
                    "then open a NEW terminal — winget's PATH change does not reach "
                    "shells that are already running.")
        sys.exit(f"Error: {' and '.join(missing)} not found on PATH.\n{hint}")


def clean_arg_path(s: str) -> str:
    """Windows shells hand us `C:\\Library"` when the user tab-completes a directory and
    types `"C:\\Library\\"` — the trailing backslash escapes the closing quote. A double
    quote is not a legal character in a Windows path, so stripping it is always safe."""
    return s.rstrip('"') if sys.platform == "win32" else s


def main():
    global VERBOSE, COND_ROOT
    ap = argparse.ArgumentParser(description="Condensed audio + Japanese subtitles from sub-timed audio (anime-whisper)")
    ap.add_argument("inputs", nargs="+", help="video file(s), or a directory with --batch")
    ap.add_argument("--batch", action="store_true", help="treat the single argument as a directory")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without doing work")
    ap.add_argument("--no-subs", action="store_true", help="skip subtitle generation")
    ap.add_argument("--no-condensed", action="store_true", help="skip condensed-audio generation")
    ap.add_argument("--quiet", action="store_true", help="less verbose output")
    ap.add_argument("--condensed-dir", metavar="DIR",
                    help="where condensed audio goes; <DIR>/<series>/Condensed Audio/ "
                         "(default: your Music folder, so media servers don't index it)")
    args = ap.parse_args()
    if args.condensed_dir:
        COND_ROOT = clean_arg_path(args.condensed_dir)
    args.inputs = [clean_arg_path(s) for s in args.inputs]

    VERBOSE = not args.quiet
    want_subs = not args.no_subs
    want_cond = not args.no_condensed
    if not want_subs and not want_cond:
        sys.exit("Nothing to do: both --no-subs and --no-condensed given.")

    check_tools()

    if args.dry_run:
        print("(dry-run: nothing will be written)\n")

    try:
        if args.batch or (len(args.inputs) == 1 and Path(args.inputs[0]).is_dir()):
            directory = Path(args.inputs[0])
            if not directory.is_dir():
                sys.exit(f"Error: directory not found: {directory}")
            files = collect_videos(directory)
            if not files:
                print(f"No video files {VIDEO_EXTS} found under: {directory}")
                return
            print(f"Library scan: {len(files)} video file(s) under {directory}")
            print(f"Producing: {' + '.join((['condensed audio'] if want_cond else []) + (['subtitles'] if want_subs else []))}\n")
            tally = {"ok": 0, "skipped": 0, "failed": 0, "missing": 0, "nosubs": 0}
            for f in files:
                tally[process_one(f, directory, True, args.dry_run, want_subs, want_cond)] += 1
            print(f"=== Library complete: {tally['ok']} done, {tally['skipped']} skipped, "
                  f"{tally['nosubs']} no-subs, {tally['failed'] + tally['missing']} failed (of {len(files)})")
            return

        ok = other = 0
        for name in args.inputs:
            v = Path(name)
            result = process_one(v, v.parent, False, args.dry_run, want_subs, want_cond)
            ok += result == "ok"
            other += result != "ok"
        if len(args.inputs) > 1:
            print(f"=== Done: {ok} done, {other} not done (of {len(args.inputs)})")
    except KeyboardInterrupt:
        print("\n\nInterrupted (Ctrl-C) — stopping. Finished files are kept; the "
              "in-progress file was discarded. Re-run the same command to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()

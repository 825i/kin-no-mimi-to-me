#!/usr/bin/env python3
"""
anime-jp-immersion — subtitle-timed anime-whisper + condensed audio.

For each episode it uses an existing subtitle track's TIMESTAMPS as the timing oracle
(human-made, frame-accurate) and then does two things off those same timings:

  1. CONDENSED AUDIO — cuts the dialogue segments out of the original audio and
     concatenates them into one small file (silence/music/gaps removed), the way impd
     does, with metadata so it shows up properly in music/audio players. For listening
     immersion.  ->  <series dir>/Condensed Audio/<name>.ogg
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
  JPSUBS_MODEL   Hugging Face model id to use (default: litagin/anime-whisper).
                 Point this at an updated/alternative model without touching the code.
  JPSUBS_DEVICE  Force cuda | mps | cpu (default: auto-detect).

Requirements: Python 3.9+, ffmpeg on PATH, and `pip install -r requirements.txt`.

Usage (via the `jpsubs` launcher, or `python subgen.py`):
    jpsubs [options] --batch <directory>
    jpsubs [options] <video> [<video> ...]
Options: --dry-run, --no-subs, --no-condensed, --quiet
"""
import argparse
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

# ---- Configuration (env-overridable where it matters) ----------------------
MODEL_ID = os.environ.get("JPSUBS_MODEL", "litagin/anime-whisper")
AUDIO_TRACK = 0                              # which audio stream to use (0 = first)
PREF_SUB_LANG = "en"                         # preferred embedded subtitle language
DENOISE_FILTER = "afftdn=nr=12"              # light denoise for the ASR audio; "" to disable
SEG_PAD_S = 0.20                             # ASR: audio padding each side of a segment
MIN_SEG_S = 0.20                             # ignore sub segments shorter than this
NO_REPEAT_NGRAM = 5                          # anime-whisper loop guard (per model card)
SR = 16000                                   # ASR sample rate

# Condensed audio is produced by the REAL impd (github.com/Ajatt-Tools/impd), vendored at
# ./vendor/impd and invoked directly — no reimplementation. impd requires Bash 5+ and GNU
# tools (macOS: `brew install bash grep findutils coreutils`). macOS/Linux only.
COND_DIR = "Condensed Audio"                 # condensed files go in this folder in the series dir
COND_FILE_EXT = ".ogg"
COND_BITRATE = "32k"                         # passed to impd's config; opus VBR is plenty for speech
IMPD = Path(__file__).resolve().parent / "vendor" / "impd"

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


def cond_output_for(video: Path, root: Path) -> Path:
    """Condensed audio path: <series>/<COND_DIR>/<name>.ogg.
    Uses the episode-folder name for per-episode layouts (more meaningful and avoids
    collisions when inner files are generically named), else the video's own name."""
    sdir = series_dir_for(video, root)
    base = video.with_suffix("").name if video.parent == sdir else video.parent.name
    return sdir / COND_DIR / (base + COND_FILE_EXT)


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
    series = sdir.name
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
    """Locate a Bash >= 5 (impd requires it). macOS system bash is 3.2; Homebrew provides 5."""
    global _BASH5
    if _BASH5 == "?":
        _BASH5 = None
        seen = []
        for c in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", shutil.which("bash"), "/bin/bash"):
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


def make_condensed(video: Path, timing_srt: Path, out_path: Path, meta: dict):
    """Make the condensed audio with the REAL impd (vendored at ./vendor/impd), then add
    player metadata (impd strips tags). impd cuts each dialogue line independently, so the
    result is clean — no boundary stutter. Requires Bash 5+ and GNU tools; if they're
    missing the condensed step is skipped (subtitles still work)."""
    bash5 = find_bash5()
    if bash5 is None or not IMPD.exists():
        vprint("    Condensed audio skipped: needs Bash 5 + impd "
               "(macOS: brew install bash grep findutils coreutils).")
        return
    with tempfile.TemporaryDirectory(prefix="subgen.impd.") as td:
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
        # Add metadata via stream copy (no re-encode), written atomically.
        tmp_out = out_path.parent / (out_path.name + ".part.ogg")
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
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    vprint(f"    -> {out_path.name}  ({size_mb:.1f} MB)")


# ---- ASR (anime-whisper) ---------------------------------------------------
def extract_asr_audio(video: Path, wav: Path):
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-map", f"0:a:{AUDIO_TRACK}"]
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
        _PIPE = pipeline("automatic-speech-recognition", model=MODEL_ID,
                         device=device, dtype=dtype, chunk_length_s=30.0, batch_size=1)
    return _PIPE


def transcribe_span(pipe, audio, start, end):
    a = max(0, int((start - SEG_PAD_S) * SR))
    b = min(len(audio), int((end + SEG_PAD_S) * SR))
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
    tmp = path.parent / (path.name + ".part")  # atomic: write then rename
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for i, e in enumerate(entries, 1):
                f.write(f"{i}\n{fmt_ts(e['start'])} --> {fmt_ts(e['end'])}\n{e['text']}\n\n")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_subtitles(video: Path, spans, out_path: Path):
    with tempfile.TemporaryDirectory(prefix="subgen.asr.") as td:
        wav = Path(td) / "audio.wav"
        note = f"denoise {DENOISE_FILTER}" if DENOISE_FILTER else "no denoise"
        vprint(f"    Extracting ASR audio (16kHz mono, {note})...")
        extract_asr_audio(video, wav)
        audio = load_wav_f32(wav)
        spans = dedup_spans(spans)  # collapse overlapping karaoke/duplicate windows
        pipe = get_pipe()
        vprint("")  # separate the model-loading progress bar from the transcription
        vprint(f"    Transcribing {len(spans)} segments with anime-whisper:")
        entries = []
        for start, end in spans:
            text = transcribe_span(pipe, audio, start, end)
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

    try:
        with tempfile.TemporaryDirectory(prefix="subgen.") as td:
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
                    vprint(f"    [dry-run] subtitles      -> {sub_out}")
                print(f"    [dry-run] from {len(spans)} segments\n")
                return "ok"

            if do_cond:
                vprint("")
                vprint("    Condensing with impd...")
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
        sys.exit(f"Error: {' and '.join(missing)} not found on PATH. "
                 "Install ffmpeg — see https://ffmpeg.org/download.html")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(description="Condensed audio + Japanese subtitles from sub-timed audio (anime-whisper)")
    ap.add_argument("inputs", nargs="+", help="video file(s), or a directory with --batch")
    ap.add_argument("--batch", action="store_true", help="treat the single argument as a directory")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without doing work")
    ap.add_argument("--no-subs", action="store_true", help="skip subtitle generation")
    ap.add_argument("--no-condensed", action="store_true", help="skip condensed-audio generation")
    ap.add_argument("--quiet", action="store_true", help="less verbose output")
    args = ap.parse_args()

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

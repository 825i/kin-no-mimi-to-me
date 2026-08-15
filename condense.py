#!/usr/bin/env python3
#
# Condensed audio for Windows — a port of the `condense` command from
# Immersion Pod (impd), Copyright (C) 2021-2026 Ajatt-Tools and contributors.
# https://github.com/Ajatt-Tools/impd
#
# This file was written by translating impd's condense logic (vendored at ./vendor/impd)
# step for step, so it is a derivative work of impd and is distributed under impd's
# licence, the GNU General Public License v3 or later — NOT the MIT licence that covers
# the rest of this project.
#
# IMPORTANT: subgen.py runs this as a SEPARATE PROGRAM (subprocess), never as an import,
# for exactly the reason the project already invokes vendor/impd as a separate program
# rather than linking it. Please keep it that way: importing this module into the
# MIT-licensed code would combine the two into one work and drag the whole thing under
# the GPL. Nothing here is imported by subgen.py, and nothing here imports subgen.py.
#
# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
# <https://www.gnu.org/licenses/>
#
# Not affiliated with or endorsed by Ajatt-Tools.
#
"""Native condensed-audio generator — a faithful port of impd's `condense` command.

WHY THIS EXISTS
---------------
Condensed audio is normally produced by the real impd (vendored at ./vendor/impd), which
is a Bash 5 + GNU-tools program. That works well on macOS and Linux. It cannot work on
Windows, even under Git Bash / MSYS2, for three independent reasons:

  1. impd writes an ffmpeg concat list holding POSIX paths (/tmp/immersionpod/...). The
     list is read by ffmpeg itself, so MSYS's argv path translation never applies, and
     native ffmpeg.exe resolves "/tmp/..." as "C:/tmp/..." -> "Impossible to open".
     impd then falls through to `add_metadata` on the UNcondensed temp audio and exits 0,
     so the caller happily accepts a full-length file as "condensed audio".
  2. impd's canonicalize() only recognises paths starting with "/", so a Windows path is
     treated as relative and gets $PWD prepended ("/c/Users/x/C:\\Users\\...").
  3. MSYS converts argv to the ANSI codepage when invoking native binaries, mangling
     non-ASCII filenames. Japanese episode filenames simply fail to open.

So on Windows we do the same work in Python, driving ffmpeg directly. The algorithm below
mirrors impd v0.10 step for step — same track selection weights, same padding, same chunk
merging, same numeric formatting, same opus encoder flags — so output matches what the
macOS/Linux impd path produces.

Deliberate divergences from impd, all of them fixes rather than changes of behaviour:
  * If no usable timings are found we raise instead of emitting a full-length file. impd's
    awk unconditionally prints a final "0,0" record and its shell falls back to the
    uncondensed audio; that silent fallback is exactly the corruption described above.
  * We verify the concatenated result is meaningfully shorter than the source audio, and
    raise if it is not. Belt and braces against the same class of failure.
  * Chapter titles are read from ffprobe JSON rather than CSV, so a title containing a
    comma cannot shift the fields it is matched against.
  * The external-subtitle search is sorted (deterministic rather than filesystem order)
    and skips our own ".ja.srt" output so a previous run can never become the timing
    source for a later one.
  * impd's line_skip_pattern contains ".*{\\be1}.*"; awk reads "\\b" as a backspace
    character, so that alternative can never match real subtitle text. We match the
    literal "{\\be1}" that was clearly intended (an ASS blur override tag).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- impd's config defaults (vendor/impd: set_config_defaults) --------------
LANGS = "japanese,jpn,jp,ja,english,eng,en,russian,rus,ru"
PREFER_INTERNAL_SUBS = True
MAX_CHUNK_LEN_S = 30.0
PADDING = 0.2
IGNORED_CHAPTERS_PATTERN = "PV|OP|ED|Intro"
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"

# impd's line_skip_pattern: music-note-only lines carry no dialogue.
_LINE_SKIP_RE = re.compile(r"^♬〜$|^♪?〜♪?$|^・～$|.*\{\\be1\}.*")

_SRT_TIMING_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})$")

# On Windows, keep every ffmpeg/ffprobe child out of the user's face.
_NO_WINDOW = {}
if sys.platform == "win32":
    _NO_WINDOW = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


class CondenseError(RuntimeError):
    pass


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", **_NO_WINDOW, **kw)


def _mmss(sec: float) -> str:
    """mm:ss.s — floor the minutes; a plain %.0f would round 33.2s up to '01:33.2'."""
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def _fmt(v: float) -> str:
    """awk's default output format for numbers is %.6g; impd's chunk timestamps go
    through it on their way to ffmpeg, so we format identically."""
    return f"{v:.6g}"


# ---- track selection (vendor/impd: probe_tracks / guess_track_priority) -----
def probe_tracks(video: Path):
    """Absolute stream index, language, title and type for every A/V/S stream."""
    cp = _run(["ffprobe", "-loglevel", "error", "-show_entries",
               "stream=index,codec_type:stream_tags=language,title",
               "-of", "json", str(video)])
    if cp.returncode != 0:
        raise CondenseError(f"ffprobe failed on {video.name}: {cp.stderr.strip()[:200]}")
    try:
        streams = json.loads(cp.stdout).get("streams", [])
    except json.JSONDecodeError as e:
        raise CondenseError(f"could not parse ffprobe output: {e}")
    out = []
    for s in streams:
        if s.get("codec_type") not in ("audio", "subtitle", "video"):
            continue
        tags = s.get("tags", {}) or {}
        out.append({
            "index": s.get("index"),
            # impd's awk defaults missing tags to the string "unknown"; the weighting
            # below depends on that, so reproduce it exactly.
            "lang": (tags.get("language") or "unknown"),
            "title": (tags.get("title") or "unknown"),
            "type": s.get("codec_type"),
        })
    return out


def guess_track_priority(track_lang: str, track_title: str) -> int:
    """impd's weighting. LOWER is better; impd sorts ascending and takes the first."""
    lang, title = track_lang.lower(), track_title.lower()

    # Bash `case` takes the first matching branch, so *full* beats *song* etc.
    if "full" in title:
        weight = 100
    elif any(t in title for t in ("song", "sign", "caption", "comment", "forced")):
        weight = 900
    else:
        weight = 500

    prefs = LANGS.lower().split(",")
    for pref in prefs:                       # penalise non-preferred languages
        if lang == pref:
            break
        weight += 10
    for pref in prefs:                       # some containers put the language in title
        if pref in title:
            break
        weight += 1
    return weight


def best_track(kind: str, video: Path):
    """Absolute stream index of the best 'audio' or 'subtitle' track, or None.
    Stable sort on weight, first wins — same as impd's `sort -s -g -k 2 | head -1`."""
    cands = [t for t in probe_tracks(video) if t["type"] == kind]
    if not cands:
        return None
    scored = [(guess_track_priority(t["lang"], t["title"]), i, t)
              for i, t in enumerate(cands)]
    scored.sort(key=lambda x: (x[0], x[1]))   # weight, then original order (stable)
    return scored[0][2]["index"]


# ---- audio extraction (vendor/impd: extract_audio) -------------------------
def _is_ogg_audio(path: Path) -> bool:
    cp = _run(["ffprobe", "-loglevel", "error", "-show_entries",
               "format=format_name:stream=codec_type", "-of", "json", str(path)])
    if cp.returncode != 0:
        return False
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return False
    fmt = (data.get("format", {}) or {}).get("format_name", "")
    types = {s.get("codec_type") for s in data.get("streams", [])}
    return "ogg" in fmt and "video" not in types


def extract_audio(video: Path, out_ogg: Path, bitrate: str, track_index=None):
    """Full-length opus of the chosen audio track, loudness-normalised. Same encoder
    flags and same order as impd, so the bitstream matches."""
    if track_index is None:
        track_index = best_track("audio", video)
    mapping = f"0:{track_index}" if track_index is not None else "0:a:0"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-hide_banner", "-y",
           "-vn", "-sn", "-i", str(video), "-map_metadata", "-1", "-map", mapping,
           "-ac", "2", "-ab", bitrate, "-vbr", "on", "-compression_level", "10",
           "-application", "voip", "-acodec", "libopus",
           "-af", LOUDNORM, str(out_ogg)]
    cp = _run(cmd)
    if cp.returncode != 0 or not out_ogg.exists() or out_ogg.stat().st_size == 0:
        raise CondenseError(f"audio extraction failed: {cp.stderr.strip()[:300]}")
    return mapping


# ---- subtitle acquisition (vendor/impd: fetch_video_subtitles) --------------
def extract_subtitles(video: Path, out_srt: Path, track_index=None) -> bool:
    if track_index is None:
        track_index = best_track("subtitle", video)
    if track_index is None:
        return False
    cp = _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-hide_banner", "-y",
               "-vn", "-an", "-i", str(video), "-map_metadata", "-1",
               "-map", f"0:{track_index}", "-f", "srt", str(out_srt)])
    return cp.returncode == 0 and out_srt.exists() and out_srt.stat().st_size > 0


def find_external_subtitles(video: Path, exclude_suffix=".ja.srt"):
    """First .srt/.ass alongside the video whose name contains the episode stem.
    Sorted for determinism; our own generated subtitles are never eligible."""
    stem = video.with_suffix("").name.lower()
    try:
        entries = sorted(video.parent.iterdir())
    except OSError:
        return None
    for p in entries:
        if not p.is_file() or p.suffix.lower() not in (".srt", ".ass"):
            continue
        if exclude_suffix and p.name.endswith(exclude_suffix):
            continue
        if stem in p.name.lower():
            return p
    return None


def sub_conv(src: Path, out_srt: Path) -> bool:
    cp = _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-hide_banner", "-y",
               "-vn", "-an", "-i", str(src), "-f", "srt", str(out_srt)])
    return cp.returncode == 0 and out_srt.exists() and out_srt.stat().st_size > 0


def fetch_video_subtitles(video: Path, out_srt: Path, track_index=None) -> str:
    """Returns a short description of where the timings came from."""
    external = find_external_subtitles(video)
    if PREFER_INTERNAL_SUBS:
        if extract_subtitles(video, out_srt, track_index):
            return "embedded subtitle track"
        if external and sub_conv(external, out_srt):
            return f"external '{external.name}'"
    else:
        if external and sub_conv(external, out_srt):
            return f"external '{external.name}'"
        if extract_subtitles(video, out_srt, track_index):
            return "embedded subtitle track"
    raise CondenseError("no usable subtitle track for condensing")


# ---- timing maths (vendor/impd: filter_non_speech_fragments + parse_speech_fragments)
def _srt_time_to_seconds(t: str) -> float:
    h, m, rest = t.split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + float(rest.replace(",", "."))


def filter_non_speech_fragments(srt_text: str):
    """Return the timing lines, dropping any whose following text is a music-note-only
    line. Mirrors impd's awk: a skip-match retracts the most recently kept timing."""
    timings = []
    skip = 0
    for raw in srt_text.splitlines():
        line = raw.rstrip("\r")
        if _SRT_TIMING_RE.match(line):
            timings.append(line)
            skip = 0
            continue
        if _LINE_SKIP_RE.match(line) and skip == 0:
            skip = 1
            if timings:
                timings.pop()
    return timings


def parse_speech_fragments(timing_lines):
    """Pad each span, then merge any span that overlaps the previous one at all.
    Returns [(start, end)] as floats, in order."""
    out = []
    prev = None
    for line in timing_lines:
        m = _SRT_TIMING_RE.match(line)
        if not m:
            continue
        start = _srt_time_to_seconds(m.group(1))
        end = _srt_time_to_seconds(m.group(2))
        if start == end or (end - start) > MAX_CHUNK_LEN_S:
            continue
        start = max(0.0, start - PADDING)
        end = end + PADDING
        if prev is None:
            prev = [start, end]
            continue
        # impd: overlap() > 0, measured as a fraction of the previous span's length.
        overlap = min(prev[1], end) - max(prev[0], start)
        if overlap > 0:
            prev = [min(prev[0], start), max(prev[1], end)]
        else:
            out.append((prev[0], prev[1]))
            prev = [start, end]
    if prev is not None:
        out.append((prev[0], prev[1]))
    return out


# ---- ignored chapters (vendor/impd: fetch_ignored_chapters/filter_ignored_chapters)
def fetch_ignored_chapters(video: Path, pattern: str):
    if not pattern:
        return []
    cp = _run(["ffprobe", "-loglevel", "error", "-show_chapters",
               "-of", "json", str(video)])
    if cp.returncode != 0:
        return []
    try:
        chapters = json.loads(cp.stdout).get("chapters", [])
    except json.JSONDecodeError:
        return []
    rx = re.compile(pattern)          # unanchored + case-sensitive, as impd has it
    out = []
    for ch in chapters:
        title = ((ch.get("tags", {}) or {}).get("title") or "")
        if rx.search(title):
            try:
                out.append((float(ch["start_time"]), float(ch["end_time"])))
            except (KeyError, TypeError, ValueError):
                pass
    return out


def filter_ignored_chapters(spans, ignored):
    if not ignored:
        return list(spans)
    return [(s, e) for s, e in spans
            if not any(s < ie and e > istart for istart, ie in ignored)]


# ---- chunking + concat (vendor/impd: make_chunk / concat_audio) -------------
def make_chunk(src_ogg: Path, out_ogg: Path, start: str, end: str) -> bool:
    cp = _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-hide_banner", "-y",
               "-vn", "-sn", "-i", str(src_ogg), "-map_metadata", "-1",
               "-codec:a", "copy", "-ss", start, "-to", end, str(out_ogg)])
    return cp.returncode == 0 and out_ogg.exists() and out_ogg.stat().st_size > 0


def concat_audio(list_file: Path, out_ogg: Path):
    cp = _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-hide_banner", "-y",
               "-f", "concat", "-safe", "0", "-i", str(list_file),
               "-map_metadata", "-1", "-c", "copy", str(out_ogg)])
    if cp.returncode != 0 or not out_ogg.exists() or out_ogg.stat().st_size == 0:
        raise CondenseError(f"concat failed: {cp.stderr.strip()[:300]}")


def duration_of(path: Path):
    cp = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(path)])
    try:
        return float(cp.stdout.strip())
    except ValueError:
        return None


# ---- the public entry point ------------------------------------------------
def condense(video: Path, out_ogg: Path, bitrate: str = "32k", log=None):
    """Write condensed dialogue-only audio for `video` to `out_ogg`.

    Raises CondenseError on any failure — notably, it never falls back to writing
    full-length audio, which is what impd does on Windows.
    """
    def say(msg):
        if log:
            log(msg)

    video, out_ogg = Path(video), Path(out_ogg)
    if not video.is_file():
        raise CondenseError(f"not a file: {video}")

    with tempfile.TemporaryDirectory(prefix="subgen.cond.") as td:
        tdp = Path(td)
        temp_audio = tdp / "audio.ogg"
        subs = tdp / "timing.srt"

        if _is_ogg_audio(video):
            # impd copies ogg audio through untouched; nothing to condense against.
            raise CondenseError("input is already an ogg audio file, not a video")

        mapping = extract_audio(video, temp_audio, bitrate)
        source = fetch_video_subtitles(video, subs)

        srt_text = subs.read_text(encoding="utf-8", errors="replace")
        spans = parse_speech_fragments(filter_non_speech_fragments(srt_text))
        ignored = fetch_ignored_chapters(video, IGNORED_CHAPTERS_PATTERN)
        spans = filter_ignored_chapters(spans, ignored)
        if not spans:
            raise CondenseError("no dialogue spans found — refusing to write "
                                "full-length audio")
        say(f"      timing: {source}, audio stream {mapping}, "
            f"{len(spans)} chunk(s)"
            + (f", {len(ignored)} ignored chapter(s)" if ignored else ""))

        # Chunks live next to the list file and are referenced by bare relative names,
        # so no path escaping is needed inside the concat list (backslashes in Windows
        # paths would otherwise be read as escape characters by the concat demuxer).
        chunks_dir = tdp / "chunks"
        chunks_dir.mkdir()
        kept, total_cut = [], 0.0
        for i, (start, end) in enumerate(spans):
            s, e = _fmt(start), _fmt(end)
            name = f"chunk_{i:05d}.ogg"
            if make_chunk(temp_audio, chunks_dir / name, s, e):
                kept.append(name)
                total_cut += end - start
                say(f"      cut [{_mmss(start)} → {_mmss(end)}]  {end - start:4.1f}s")
        if not kept:
            raise CondenseError("every chunk failed to cut")

        list_file = chunks_dir / "chunks.list"
        with open(list_file, "w", encoding="utf-8", newline="\n") as f:
            for name in kept:
                f.write(f"file '{name}'\n")

        staged = tdp / "condensed.ogg"
        concat_audio(list_file, staged)

        # Guard against the impd-on-Windows failure mode: a "condensed" file that is
        # really just the whole episode.
        src_dur, out_dur = duration_of(temp_audio), duration_of(staged)
        if src_dur and out_dur and out_dur > 0.9 * src_dur and src_dur - total_cut > 5:
            raise CondenseError(
                f"condensed output is {out_dur:.1f}s against a {src_dur:.1f}s source "
                f"but only {total_cut:.1f}s of dialogue was cut — refusing it")

        out_ogg.parent.mkdir(parents=True, exist_ok=True)
        # Written straight to the requested path. subgen.py always points this at a
        # throwaway file and does its own atomic tag-and-rename into the library, so
        # nothing half-written can appear there.
        try:
            os.replace(staged, out_ogg)
        except OSError:
            shutil.move(str(staged), str(out_ogg))   # different volume
        return {"chunks": len(kept), "dialogue_s": total_cut,
                "source_s": src_dur, "out_s": out_dur, "timing_source": source}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="impd-compatible condensed audio (native). Invoked as a separate "
                    "program by subgen.py; also usable on its own.")
    ap.add_argument("video")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--bitrate", default="32k")
    ap.add_argument("--json-out", metavar="PATH",
                    help="write the result summary to PATH as JSON (used by subgen.py, "
                         "so the two never have to share a process)")
    ap.add_argument("--quiet", action="store_true", help="no per-chunk progress")
    ap.add_argument("--chunks-only", action="store_true",
                    help="print the computed chunk list and exit (for testing)")
    a = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    if a.chunks_only:
        with tempfile.TemporaryDirectory() as td:
            srt = Path(td) / "t.srt"
            fetch_video_subtitles(Path(a.video), srt)
            text = srt.read_text(encoding="utf-8", errors="replace")
            for s, e in parse_speech_fragments(filter_non_speech_fragments(text)):
                print(f"{_fmt(s)},{_fmt(e)}")
        return 0

    # flush=True so a caller streaming our stdout sees progress as it happens
    log = None if a.quiet else (lambda m: print(m, flush=True))
    try:
        info = condense(Path(a.video), Path(a.output), a.bitrate, log=log)
    except CondenseError as e:
        print(f"condense failed: {e}", file=sys.stderr, flush=True)
        return 1
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(info), encoding="utf-8")
    else:
        print(f"-> {a.output}  {info['chunks']} chunks, "
              f"{info['dialogue_s']:.1f}s of {info['source_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

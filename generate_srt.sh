#!/usr/bin/env bash
# OPTIONAL FALLBACK — pure-audio transcription for videos that have NO subtitles.
#
# The main tool is subgen.py (subtitle-timed anime-whisper). Use THIS script only for
# files with no embedded/external subtitle track to time against. It requires whisper.cpp
# (`whisper-cli`) and the large-v3 + Silero VAD ggml models installed separately — see the
# README. Primarily tested on macOS (Metal); it also runs on Linux with whisper.cpp built.
#
# Generate Japanese SRT subtitles from video audio using whisper.cpp (Metal GPU accelerated).
#
# Transcribes the SPOKEN Japanese audio directly with Whisper (large-v3). It does NOT
# translate or read any embedded subtitle track. Each <video>.<ext> produces <video>.srt
# right next to it, in the same folder, with the same base name.
#
# Tuned for ACCURACY over speed: full large-v3 model, widened beam search, temperature
# fallback on, plus an automatic hallucination/timing cleanup pass.
#
# VAD (Silero voice-activity detection) is ON. It gates Whisper to detected speech so
# the model can't hallucinate stock phrases like "ご視聴ありがとうございました" over
# quiet intros/music — a notorious failure mode that also derails real dialogue via
# context carryover. Set USE_VAD=0 to transcribe the full audio (captures sung lyrics
# too, but hallucinates over non-speech). VAD_THRESHOLD tunes sensitivity.
#
# Resilience: batch mode skips episodes that already have an .srt (so you can resume),
# and one failed file no longer aborts the whole run.

set -uo pipefail   # NOTE: intentionally no `-e`; errors are handled explicitly so a
                   # single bad file can't silently kill the whole batch.

# ---- Configuration ---------------------------------------------------------
MODEL="$HOME/.cache/whisper-cpp/ggml-large-v3.bin"
VAD_MODEL="$HOME/.cache/whisper-cpp/ggml-silero-v6.2.0.bin"
PYTHON="/opt/homebrew/bin/python3.12"
WHISPER_LANG="ja"          # force Japanese; never auto-detect
AUDIO_TRACK=0              # first audio track
DENOISE_FILTER="afftdn=nr=12"   # ffmpeg -af applied before transcription to tame tape/VHS
                                # hiss (measurably helped on this noisy 1986 source). Keep it
                                # LIGHT — aggressive filtering adds artifacts that hurt. Empty = off.
BEAM=8                     # beam size (default 5) — wider = more thorough search
BEST_OF=8                  # candidates kept for temperature-fallback sampling
USE_VAD=1                  # 1 = Silero VAD gates to speech (prevents non-speech hallucination)
VAD_THRESHOLD=0.5          # VAD sensitivity 0-1: lower keeps more (quiet) speech + more risk of
                           # transcribing music/noise; higher is stricter. 0.5 is a good default.
THREADS="$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
VIDEO_EXTS=(mp4 mkv mov m4v avi ts webm)   # extensions picked up in --batch mode
DRY_RUN=0                  # set via --dry-run: preview the plan without transcribing

usage() {
    cat <<EOF
Usage: $0 [--dry-run] <video_file> [video_file2 ...]
       $0 [--dry-run] --batch <directory>

Generates Japanese .srt subtitles from the spoken audio using whisper.cpp (large-v3).
Each <video> yields <video>.srt in the same folder, same base name.

  --batch <dir>   Process every video in <dir>. Skips files that already have an .srt
                  (resume-friendly) and continues past any file that fails.
  --dry-run       Show exactly what would be processed/skipped, without transcribing.
EOF
    exit 1
}

# ---- SRT post-processing (hallucination loops + timing) --------------------
clean_srt() {
    local srt_file="$1"
    "$PYTHON" - "$srt_file" << 'PYTHON_EOF'
import re, sys
from pathlib import Path

def parse_srt(text):
    blocks = re.split(r'\n\n+', text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        m = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        end = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        entries.append({'start': start, 'end': end, 'text': '\n'.join(lines[2:]).strip()})
    return entries

def fmt(s):
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    sec = int(r)
    ms = int((r - sec) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{sec:02d},{ms:03d}"

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding='utf-8')
except UnicodeDecodeError:
    text = path.read_text(encoding='utf-8', errors='replace')

entries = parse_srt(text)
orig = len(entries)

# Pass 1: Remove consecutive duplicate lines (hallucination loops)
cleaned = []
repeat = 0
for e in entries:
    t = e['text'].strip()
    if not t:
        continue
    if cleaned and t == cleaned[-1]['text']:
        repeat += 1
        if repeat <= 2:
            cleaned.append(e)
        continue
    repeat = 0
    cleaned.append(e)

# Pass 2: Remove globally over-represented phrases (hallucination)
counts = {}
for e in cleaned:
    counts[e['text']] = counts.get(e['text'], 0) + 1
total = len(cleaned)
bad = {t for t, c in counts.items() if c > 5 and c / total > 0.05}
if bad:
    for b in bad:
        print(f"    Removed hallucination: '{b}' ({counts[b]}x)")
    cleaned = [e for e in cleaned if e['text'] not in bad]

# Pass 2b: Remove known Whisper hallucination phrases (fabricated over non-speech).
# These YouTube-style outros never occur in broadcast dialogue, so a substring match
# is safe and catches residuals that slip past VAD.
MARKERS = ('ご視聴ありがとう', 'ご視聴いただき', 'ご清聴ありがとう', 'チャンネル登録',
           'グッドボタン', 'ご覧いただきありがとう', '高評価')
hits = [e['text'] for e in cleaned if any(m in e['text'] for m in MARKERS)]
if hits:
    print(f"    Removed {len(hits)} known-hallucination line(s) (e.g. '{hits[0]}')")
    cleaned = [e for e in cleaned if not any(m in e['text'] for m in MARKERS)]

# Pass 3: Cap duration to 7s and ensure gaps between subtitles
for i, e in enumerate(cleaned):
    if e['end'] - e['start'] > 7.0:
        e['end'] = e['start'] + 7.0
    if i + 1 < len(cleaned):
        nxt = cleaned[i+1]['start']
        if e['end'] > nxt - 0.1:
            e['end'] = max(e['start'] + 0.5, nxt - 0.1)

# Pass 4: Remove entries shorter than 0.3s (noise)
cleaned = [e for e in cleaned if e['end'] - e['start'] >= 0.3]

with open(path, 'w', encoding='utf-8') as f:
    for i, e in enumerate(cleaned, 1):
        f.write(f"{i}\n{fmt(e['start'])} --> {fmt(e['end'])}\n{e['text']}\n\n")

removed = orig - len(cleaned)
if removed > 0:
    print(f"    Cleaned: {removed} entries removed ({orig} -> {len(cleaned)})")
else:
    print(f"    Clean: {len(cleaned)} entries, no issues found")
PYTHON_EOF
}

# ---- Transcribe a single file ---------------------------------------------
# Returns: 0 = success, 1 = failure. Never aborts the caller.
transcribe_file() {
    local input="$1"
    local base="${input%.*}"
    local output_srt="${base}.srt"

    if [ "$DRY_RUN" = "1" ]; then
        echo "=== [dry-run] Would transcribe: $(basename "$input")"
        echo "    -> $output_srt"
        echo ""
        return 0
    fi

    local tmpdir
    tmpdir="$(mktemp -d "/tmp/whisper.XXXXXX")" || { echo "    ERROR: could not create temp dir"; return 1; }
    local tmpwav="$tmpdir/audio.wav"

    echo "=== Processing: $(basename "$input")"

    # Extract the chosen audio track as 16 kHz mono PCM (exactly what Whisper wants),
    # optionally applying a light denoise filter first. Built as an array so the filter
    # can be omitted cleanly (safe under `set -u`).
    local -a ffargs=(-hide_banner -loglevel error -y -i "$input" -map "0:a:${AUDIO_TRACK}")
    local denoise_note="no denoise"
    if [ -n "$DENOISE_FILTER" ]; then
        ffargs+=(-af "$DENOISE_FILTER")
        denoise_note="denoise: $DENOISE_FILTER"
    fi
    ffargs+=(-ar 16000 -ac 1 -c:a pcm_s16le "$tmpwav")
    echo "    Extracting audio (track $AUDIO_TRACK, $denoise_note)..."
    if ! ffmpeg "${ffargs[@]}"; then
        echo "    ERROR: audio extraction failed"
        rm -rf "$tmpdir"
        return 1
    fi

    # Transcribe. Accuracy-first flags; GPU (Metal) is used automatically.
    # Build args as an array so VAD can be toggled cleanly (safe under `set -u`).
    local -a wargs=(-f "$tmpwav" -m "$MODEL" -l "$WHISPER_LANG" -sns
                    -bs "$BEAM" -bo "$BEST_OF" -osrt -of "$base" -t "$THREADS")
    local vad_note="off"
    if [ "$USE_VAD" = "1" ]; then
        wargs+=(--vad -vm "$VAD_MODEL" -vt "$VAD_THRESHOLD")
        vad_note="on (thold $VAD_THRESHOLD)"
    fi
    echo "    Transcribing (large-v3, beam $BEAM, VAD $vad_note, Metal GPU)..."
    if ! whisper-cli "${wargs[@]}"; then
        echo "    ERROR: transcription failed"
        rm -rf "$tmpdir"
        return 1
    fi

    rm -rf "$tmpdir"

    if [ -f "$output_srt" ]; then
        echo "    Cleaning..."
        clean_srt "$output_srt"
    else
        echo "    ERROR: no SRT file was created"
        return 1
    fi
    echo ""
    return 0
}

# ---- Process one path, honoring the skip-existing rule --------------------
# Returns: 0 ok, 1 failed, 2 missing input, 3 skipped (srt exists)
process_one() {
    local input="$1" skip_existing="$2"
    if [ ! -f "$input" ]; then
        echo "Error: $input not found — skipping"
        echo ""
        return 2
    fi
    local srt="${input%.*}.srt"
    if [ "$skip_existing" = "1" ] && [ -f "$srt" ]; then
        echo "=== Skipping (SRT already exists): $(basename "$input")"
        echo ""
        return 3
    fi
    transcribe_file "$input"
}

# ---- Dependency checks -----------------------------------------------------
if ! command -v whisper-cli &>/dev/null; then
    echo "Error: whisper-cli not found. Install with: brew install whisper-cpp"
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi
if [ "$USE_VAD" = "1" ] && [ ! -f "$VAD_MODEL" ]; then
    echo "Error: VAD model not found at $VAD_MODEL (needed because USE_VAD=1)"
    exit 1
fi
if ! "$PYTHON" --version &>/dev/null; then
    echo "Error: $PYTHON not found. Install with: brew install python@3.12"
    exit 1
fi

# ---- Parse args: pull out --dry-run, keep the rest positional -------------
_rest=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage ;;
        *) _rest+=("$a") ;;
    esac
done
if [ ${#_rest[@]} -gt 0 ]; then set -- "${_rest[@]}"; else set --; fi

[ $# -eq 0 ] && usage
[ "$DRY_RUN" = "1" ] && echo "(dry-run: nothing will be transcribed)" && echo ""

# ---- Batch mode: every video in a directory --------------------------------
if [ "${1:-}" = "--batch" ]; then
    [ $# -lt 2 ] && usage
    dir="$2"
    [ -d "$dir" ] || { echo "Error: directory not found: $dir"; exit 1; }

    # Collect video files across all known extensions, case-insensitively.
    # nullglob => unmatched patterns vanish instead of surviving literally (the old bug).
    shopt -s nullglob nocaseglob
    files=()
    for ext in "${VIDEO_EXTS[@]}"; do
        for f in "$dir"/*."$ext"; do
            files+=("$f")
        done
    done
    shopt -u nullglob nocaseglob

    if [ ${#files[@]} -eq 0 ]; then
        echo "No video files (${VIDEO_EXTS[*]}) found in: $dir"
        exit 0
    fi

    # Sort for stable, human-friendly (episode) order.
    IFS=$'\n' files=($(printf '%s\n' "${files[@]}" | sort)); unset IFS

    echo "Batch mode: ${#files[@]} video file(s) in $dir"
    echo ""

    ok=0; skipped=0; failed=0
    for f in "${files[@]}"; do
        process_one "$f" 1
        case $? in
            0) ok=$((ok + 1)) ;;
            3) skipped=$((skipped + 1)) ;;
            *) failed=$((failed + 1)); echo "    !! Error on $(basename "$f") — continuing with the rest."; echo "" ;;
        esac
    done

    echo "=== Batch complete: $ok transcribed, $skipped skipped (already had .srt), $failed failed (of ${#files[@]})"
    exit 0
fi

# ---- Single / multi-file mode (explicit files are always (re)generated) ----
ok=0; failed=0
for f in "$@"; do
    process_one "$f" 0
    case $? in
        0) ok=$((ok + 1)) ;;
        *) failed=$((failed + 1)); echo "    !! Error on $(basename "$f") — continuing."; echo "" ;;
    esac
done
[ $# -gt 1 ] && echo "=== Done: $ok transcribed, $failed failed (of $#)"

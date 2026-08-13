#!/usr/bin/env python3
"""Post-process whisper-generated SRT files to fix common issues."""

import argparse
import re
import sys
from pathlib import Path

# Known Whisper hallucination phrases (YouTube-style outros fabricated over non-speech).
# These never occur in broadcast/anime dialogue, so a substring match is safe. Shared
# with subgen.py and generate_srt.sh.
HALLUCINATION_MARKERS = (
    'ご視聴ありがとう', 'ご視聴いただき', 'ご清聴ありがとう', 'チャンネル登録',
    'グッドボタン', 'ご覧いただきありがとう', '高評価',
)


def parse_srt(text):
    """Parse SRT content into list of (index, start_sec, end_sec, text) tuples."""
    blocks = re.split(r'\n\n+', text.strip())
    entries = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        time_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})',
            lines[1]
        )
        if not time_match:
            continue
        g = [int(x) for x in time_match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = '\n'.join(lines[2:]).strip()
        entries.append({'start': start, 'end': end, 'text': text})
    return entries


def format_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{format_ts(e['start'])} --> {format_ts(e['end'])}\n")
            f.write(f"{e['text']}\n\n")


def clean_srt(entries, max_duration=7.0, min_gap=0.1, max_repeat=2):
    """
    Clean SRT entries:
    - Remove hallucination loops (same text repeated consecutively)
    - Cap subtitle duration
    - Ensure minimum gap between subtitles
    - Remove very short entries that are likely noise
    """
    cleaned = []

    # Pass 1: Remove consecutive duplicates (hallucination loops)
    repeat_count = 0
    for i, entry in enumerate(entries):
        text = entry['text'].strip()
        if not text:
            continue

        # Check if this is a repeat of the previous entry
        if cleaned and text == cleaned[-1]['text']:
            repeat_count += 1
            if repeat_count <= max_repeat:
                cleaned.append(entry)
            # else: skip this duplicate
            continue

        repeat_count = 0
        cleaned.append(entry)

    # Pass 2: Remove entries where same text appears many times total (hallucination)
    text_counts = {}
    for e in cleaned:
        text_counts[e['text']] = text_counts.get(e['text'], 0) + 1

    total = len(cleaned)
    hallucinated = set()
    for text, count in text_counts.items():
        # If a single phrase appears in >5% of all entries and more than 5 times, it's hallucination
        if count > 5 and count / total > 0.05:
            hallucinated.add(text)

    if hallucinated:
        print(f"  Detected hallucinated phrases:")
        for h in hallucinated:
            print(f"    - '{h}' (appeared {text_counts[h]} times)")
        cleaned = [e for e in cleaned if e['text'] not in hallucinated]

    # Pass 2b: Remove known Whisper hallucination phrases (fabricated over non-speech).
    # These YouTube-style outros never occur in broadcast dialogue, so a substring
    # match is safe and catches residuals that slip past VAD.
    hits = [e['text'] for e in cleaned if any(m in e['text'] for m in HALLUCINATION_MARKERS)]
    if hits:
        print(f"  Removed {len(hits)} known-hallucination line(s) (e.g. '{hits[0]}')")
        cleaned = [e for e in cleaned if not any(m in e['text'] for m in HALLUCINATION_MARKERS)]

    # Pass 3: Cap duration and ensure gaps
    for i, entry in enumerate(cleaned):
        duration = entry['end'] - entry['start']

        # Cap max duration
        if duration > max_duration:
            entry['end'] = entry['start'] + max_duration

        # Ensure this subtitle ends before the next one starts
        if i + 1 < len(cleaned):
            next_start = cleaned[i + 1]['start']
            if entry['end'] > next_start - min_gap:
                entry['end'] = max(entry['start'] + 0.5, next_start - min_gap)

    # Pass 4: Remove very short entries (< 0.3 sec) that are likely noise
    cleaned = [e for e in cleaned if e['end'] - e['start'] >= 0.3]

    return cleaned


def process_file(input_path, output_path=None):
    path = Path(input_path)
    if not path.exists():
        print(f"Error: {input_path} not found")
        return

    if output_path is None:
        output_path = path

    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        text = path.read_text(encoding='utf-8', errors='replace')
    entries = parse_srt(text)
    original_count = len(entries)

    print(f"=== Cleaning: {path.name}")
    print(f"    Original: {original_count} entries")

    cleaned = clean_srt(entries)
    final_count = len(cleaned)

    write_srt(cleaned, output_path)
    removed = original_count - final_count
    print(f"    Cleaned:  {final_count} entries ({removed} removed)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Clean whisper-generated SRT files")
    parser.add_argument("input", nargs='+', help="SRT file(s) or directory")
    parser.add_argument("--batch", action="store_true", help="Process all .srt files in directory")
    args = parser.parse_args()

    if args.batch or (len(args.input) == 1 and Path(args.input[0]).is_dir()):
        directory = Path(args.input[0])
        srts = sorted(directory.glob("*.srt"))
        print(f"Batch cleaning {len(srts)} SRT files in {directory}\n")
        for srt in srts:
            process_file(srt)
        print("=== Batch clean complete!")
    else:
        for f in args.input:
            process_file(f)


if __name__ == "__main__":
    main()

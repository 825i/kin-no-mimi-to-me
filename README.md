# 金の耳と目 (Golden Ears and Eyes)

Point it at your anime and it writes Japanese subtitles from what's actually being spoken, plus a "condensed audio" track of just the dialogue for listening practice.

It doesn't translate the existing subtitles. It uses their timing to find where each line is, then transcribes the Japanese audio itself with [anime-whisper](https://huggingface.co/litagin/anime-whisper). So the text matches the spoken words rather than an English translation.

For every video you get two files, saved next to it:

- `<episode>.ja.srt`, the Japanese subtitles.
- `<series>/Condensed Audio/<name>.ogg`, the dialogue with silence, music and gaps cut out (made by impd) and tagged so it shows up properly in a music player.

Give it one file or a whole library. It walks through every episode, skips anything it's already done, and keeps going if a file fails, so you can stop and restart whenever you like.

## Using it

You need Python 3.9+ and FFmpeg on your PATH. A GPU helps a lot but isn't required; it picks CUDA, then Apple's MPS, then the CPU.

Set it up:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

Install FFmpeg if you don't have it (`brew install ffmpeg`, `apt install ffmpeg`, or `winget install Gyan.FFmpeg`). The anime-whisper model (about 3 GB) downloads itself the first time you run it.

The condensed audio is made by impd, which needs Bash 5 and a few GNU tools. On macOS that's `brew install bash grep findutils coreutils`. If you only want subtitles you can skip all that and pass `--no-condensed`; subtitle generation has no extra requirements.

Then run it:

```bash
python subgen.py "/path/to/Episode.mkv"                 # one file
python subgen.py --batch "/path/to/Library"             # a whole library
python subgen.py --dry-run --batch "/path/to/Library"   # show what it would do first
```

Other flags: `--no-subs` (condensed audio only), `--no-condensed` (subtitles only), `--quiet`.

If you'd rather type `jpsubs` than `python subgen.py`, symlink the launcher with `ln -s "$(pwd)/subgen" /usr/local/bin/jpsubs`, or use `jpsubs.cmd` on Windows.

## Credits

This is built on other people's work. Please respect their licenses.

- [anime-whisper](https://huggingface.co/litagin/anime-whisper) by litagin does the transcription (MIT). It's built on [kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) (Apache-2.0) and [OpenAI Whisper](https://github.com/openai/whisper) (MIT).
- [impd](https://github.com/Ajatt-Tools/impd) by Ren Tatsumoto (Ajatt-Tools) makes the condensed audio. It's vendored unmodified in `vendor/impd` and stays under its own GPL-3.0 license. This project isn't affiliated with or endorsed by Ajatt-Tools.
- [Transformers](https://github.com/huggingface/transformers) (Apache-2.0) and [PyTorch](https://github.com/pytorch/pytorch) (BSD-3-Clause) run the model.
- [FFmpeg](https://ffmpeg.org) handles the audio (LGPL, or GPL depending on the build).

My own code is [MIT](LICENSE), © 2026 825i.

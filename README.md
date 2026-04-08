# REBEAT

Detect repeating musical patterns and visualize them as color-coded regions on an STFT spectrogram.

## Install (uv)

```bash
uv sync
```

## Run Gradio App

```bash
uv run python app.py
```

The app lets you upload audio, listen to it, tune analysis parameters, and inspect detected repeat groups.
It also shows a second plot with the self-similarity matrix and highlighted detected diagonals.
For looped game music, REBEAT attempts to detect a stable global loop period and run heavy analysis on one loop only, then maps detected segments back across the full track with time offsets.

## Run CLI

```bash
uv run python main.py /path/to/audio.wav --output output.png --similarity-output similarity.png --autocorr-output onset_autocorr.png
```

Useful flags:

- `--min-repeat-sec 2.5`
- `--min-lag-sec 2.0`
- `--similarity-threshold 0.72`
- `--max-patterns 12`
- `--no-mfcc` to use chroma only

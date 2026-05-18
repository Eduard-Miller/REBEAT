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
When loop optimization is active, the app shows an extra spectrogram panel for the analyzed loop segment.
You can select a `pattern_id` and build a combined playback reel that concatenates all detected instances of that pattern in chronological order.

## Component Activity Analysis App

This separate Gradio app takes a full song and a component stem, finds non-silent intervals in the component using a dBFS threshold (returned as sample index pairs), then analyzes the magnitude distribution on the full-song STFT using those intervals. It also builds a median-based magnitude mask to extract a stem.

Run it:

```bash
uv run python app_component.py
```

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

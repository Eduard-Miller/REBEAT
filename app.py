from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf

from rebeat.analysis import analyze_repeats, patterns_to_table
from rebeat.viz import (
    fig_to_numpy_rgb,
    plot_analyzed_window_spectrogram,
    plot_onset_autocorrelation,
    plot_similarity_with_diagonals,
    plot_spectrogram_with_patterns,
)


def _analyze_audio(
    audio_file: str | None,
    min_repeat_sec: float,
    min_lag_sec: float,
    similarity_threshold: float,
    max_patterns: int,
    use_mfcc: bool,
    optimize_for_loops: bool,
):
    if not audio_file:
        return (
            None,
            None,
            None,
            None,
            None,
            pd.DataFrame(),
            "Upload an audio file to begin analysis.",
            "No file analyzed yet.",
            gr.update(choices=[], value=None),
            None,
            None,
        )

    path = Path(audio_file)
    result = analyze_repeats(
        audio_path=path,
        min_repeat_sec=min_repeat_sec,
        min_lag_sec=min_lag_sec,
        similarity_threshold=similarity_threshold,
        max_patterns=max_patterns,
        use_mfcc=use_mfcc,
        optimize_for_loops=optimize_for_loops,
    )
    fig = plot_spectrogram_with_patterns(
        y=result.y, sr=result.sr, patterns=result.patterns
    )
    image_spec = fig_to_numpy_rgb(fig)
    fig_ssm = plot_similarity_with_diagonals(
        similarity=result.similarity,
        diagonal_matches=result.diagonal_matches,
    )
    image_ssm = fig_to_numpy_rgb(fig_ssm)
    fig_ac = plot_onset_autocorrelation(
        onset_autocorr=result.onset_autocorr,
        sr=result.sr,
        hop_length=result.hop_length,
        best_lag=result.onset_autocorr_best_lag,
    )
    image_ac = fig_to_numpy_rgb(fig_ac)
    fig_analyzed = plot_analyzed_window_spectrogram(
        y=result.y,
        sr=result.sr,
        analyzed_duration_sec=result.loop_optimization.analyzed_duration_sec,
        loop_optimization_applied=result.loop_optimization.applied,
        patterns=result.patterns,
        hop_length=result.hop_length,
    )
    image_analyzed = fig_to_numpy_rgb(fig_analyzed)

    rows = patterns_to_table(result.patterns)
    table = pd.DataFrame(rows)
    if table.empty:
        summary = "No repeating patterns detected. Try lowering the threshold or minimum repeat duration."
    else:
        grouped = table.groupby("pattern_id").size().to_dict()
        summary = ", ".join(
            [f"pattern_{pid:02d}: {count} segments" for pid, count in grouped.items()]
        )

    pattern_choices = [f"pattern_{p.pattern_id:02d}" for p in result.patterns]
    selector_value = pattern_choices[0] if pattern_choices else None

    state = {
        "y": result.y,
        "sr": result.sr,
        "patterns": {
            p.pattern_id: [(seg.start_sec, seg.end_sec) for seg in p.segments]
            for p in result.patterns
        },
    }

    loop = result.loop_optimization
    if loop.applied:
        loop_note = (
            "Loop optimization applied. "
            f"Estimated loop duration: {loop.loop_duration_sec:.2f}s, "
            f"estimated loops: {loop.loop_count}, "
            f"analyzed window: {loop.analyzed_duration_sec:.2f}s."
        )
    else:
        loop_note = "No stable global loop structure detected; analyzed full track."

    return (
        image_spec,
        image_analyzed,
        image_ssm,
        image_ac,
        table,
        summary,
        loop_note,
        gr.update(choices=pattern_choices, value=selector_value),
        state,
        None,
    )


def _build_pattern_reel(pattern_label: str | None, state: dict | None):
    if not state or not pattern_label:
        return None, "Run analysis and select a pattern first."

    try:
        pattern_id = int(pattern_label.split("_")[-1])
    except (ValueError, IndexError):
        return None, "Invalid pattern id selection."

    patterns = state.get("patterns", {})
    segments = patterns.get(pattern_id)
    if not segments:
        return None, "Selected pattern has no segments."

    y = state["y"]
    sr = int(state["sr"])
    clips: list[np.ndarray] = []
    gap = np.zeros(int(0.15 * sr), dtype=np.float32)
    sorted_segments = sorted(segments, key=lambda x: x[0])
    for start_sec, end_sec in sorted_segments:
        start = max(0, int(round(start_sec * sr)))
        end = min(len(y), int(round(end_sec * sr)))
        if end <= start:
            continue
        clips.append(y[start:end])
        clips.append(gap)

    if not clips:
        return None, "Could not extract valid clips for selected pattern."

    reel = np.concatenate(clips[:-1]).astype(np.float32)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, reel, sr)
        out_path = tmp.name

    return (
        out_path,
        f"Built combined reel for {pattern_label} with {len(sorted_segments)} instances.",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="REBEAT - Music Repeat Discovery") as demo:
        gr.Markdown(
            "# REBEAT\nDetect repeating music patterns and color them on an STFT spectrogram."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                min_repeat = gr.Slider(
                    0.5, 10.0, value=2.5, step=0.1, label="Minimum Repeat Length (sec)"
                )
                min_lag = gr.Slider(
                    0.5,
                    15.0,
                    value=2.0,
                    step=0.1,
                    label="Minimum Lag Between Repeats (sec)",
                )
                threshold = gr.Slider(
                    0.3, 0.98, value=0.72, step=0.01, label="Similarity Threshold"
                )
                max_patterns = gr.Slider(
                    1, 20, value=12, step=1, label="Max Pattern Groups"
                )
                use_mfcc = gr.Checkbox(
                    value=True, label="Use MFCC + Chroma (recommended)"
                )
                optimize_loops = gr.Checkbox(
                    value=True, label="Enable Loop Optimization"
                )
                run_btn = gr.Button("Run Analysis", variant="primary")

                gr.Markdown("### Pattern Reel")
                pattern_selector = gr.Dropdown(
                    choices=[],
                    label="Pattern ID",
                    value=None,
                    interactive=True,
                )
                reel_btn = gr.Button("Build Combined Pattern Track")
                reel_status = gr.Textbox(label="Pattern Reel Status")

            with gr.Column(scale=2):
                spectrogram_out = gr.Image(
                    label="Spectrogram with Pattern Highlights", type="numpy"
                )
                analyzed_window_out = gr.Image(
                    label="Analyzed Loop Segment Spectrogram", type="numpy"
                )
                similarity_out = gr.Image(
                    label="Similarity Matrix with Detected Diagonals", type="numpy"
                )
                autocorr_out = gr.Image(
                    label="Onset Autocorrelation (Loop Detection)", type="numpy"
                )
                table_out = gr.Dataframe(label="Detected Pattern Segments")
                summary_out = gr.Textbox(label="Pattern Summary")
                loop_note_out = gr.Textbox(label="Loop Optimization")
                reel_audio = gr.Audio(
                    label="Combined Pattern Instances", type="filepath"
                )

        analysis_state = gr.State(value=None)

        run_btn.click(
            fn=_analyze_audio,
            inputs=[
                audio_input,
                min_repeat,
                min_lag,
                threshold,
                max_patterns,
                use_mfcc,
                optimize_loops,
            ],
            outputs=[
                spectrogram_out,
                analyzed_window_out,
                similarity_out,
                autocorr_out,
                table_out,
                summary_out,
                loop_note_out,
                pattern_selector,
                analysis_state,
                reel_audio,
            ],
        )

        reel_btn.click(
            fn=_build_pattern_reel,
            inputs=[pattern_selector, analysis_state],
            outputs=[reel_audio, reel_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

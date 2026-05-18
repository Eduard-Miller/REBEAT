from __future__ import annotations

from dataclasses import dataclass
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import soundfile as sf

from rebeat.viz import fig_to_numpy_rgb


HOP_LENGTH = 512
N_FFT = 2048


@dataclass(frozen=True)
class IntervalResult:
    intervals: list[tuple[int, int]]
    sr: int


def _find_active_intervals_dbfs(
    y: np.ndarray,
    sr: int,
    threshold_dbfs: float,
) -> IntervalResult:
    eps = 1e-8
    dbfs = 20.0 * np.log10(np.maximum(np.abs(y), eps))
    active = dbfs >= threshold_dbfs
    if not np.any(active):
        return IntervalResult(intervals=[], sr=sr)

    changes = np.diff(active.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if active[0]:
        starts = np.concatenate(([0], starts))
    if active[-1]:
        ends = np.concatenate((ends, [len(active)]))

    intervals = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
    return IntervalResult(intervals=intervals, sr=sr)


def _intervals_to_frame_mask(
    intervals: list[tuple[int, int]],
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start_sample, end_sample in intervals:
        start_frame = max(0, int(np.floor(start_sample / hop_length)))
        end_frame = min(n_frames, int(np.ceil(end_sample / hop_length)))
        if end_frame > start_frame:
            mask[start_frame:end_frame] = True
    return mask


def _plot_histogram(mag_vals: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)
    if mag_vals.size < 2:
        ax.text(
            0.5,
            0.5,
            "No power values available for selected frames",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        return fig

    median = float(np.median(mag_vals))
    kde = gaussian_kde(mag_vals)
    x_min = float(np.min(mag_vals))
    x_max = float(np.max(mag_vals))
    if x_min == x_max:
        x_min -= 1e-6
        x_max += 1e-6
    xs = np.linspace(x_min, x_max, 512)
    ys = kde(xs)
    ax.plot(xs, ys, color="#457b9d", linewidth=1.8)
    ax.set_title("Magnitude Density (Full Song, Component Intervals)")
    ax.set_xlabel("Magnitude (|STFT|)")
    ax.set_ylabel("Probability Density")
    ax.text(
        0.98,
        0.92,
        f"median={median:.4e}",
        color="#d00000",
        fontsize=9,
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    return fig


def _plot_full_spectrogram_with_overlay(
    y: np.ndarray,
    sr: int,
    intervals: list[tuple[int, int]],
) -> plt.Figure:
    stft = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Full Song Spectrogram with Component-Driven Frame Overlay")

    for start_sample, end_sample in intervals:
        start_t = start_sample / sr
        end_t = end_sample / sr
        if end_t > start_t:
            ax.axvspan(start_t, end_t, color="#2a9d8f", alpha=0.28)
    return fig


def _plot_masked_spectrogram(
    stft_masked: np.ndarray,
    sr: int,
) -> plt.Figure:
    db = librosa.amplitude_to_db(np.abs(stft_masked), ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Masked Spectrogram (Median Magnitude Band)")
    return fig


def _plot_component_spectrogram(
    y: np.ndarray,
    sr: int,
) -> plt.Figure:
    stft = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(12, 4.0), constrained_layout=True)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Component Stem Spectrogram")
    return fig


def _analyze_component(
    full_song_path: str | None,
    component_path: str | None,
    silence_threshold_dbfs: float,
):
    if not full_song_path or not component_path:
        return (
            None,
            None,
            pd.DataFrame(),
            "Upload both the full song and component stem.",
            None,
            None,
            None,
            None,
            None,
            "Run mask extraction to generate audio outputs.",
        )

    full_path = Path(full_song_path)
    component_file = Path(component_path)

    full_y, full_sr = librosa.load(full_path.as_posix(), sr=None, mono=True)
    comp_y, comp_sr = librosa.load(component_file.as_posix(), sr=full_sr, mono=True)

    interval_result = _find_active_intervals_dbfs(
        y=comp_y,
        sr=comp_sr,
        threshold_dbfs=silence_threshold_dbfs,
    )

    intervals = interval_result.intervals
    intervals_table = pd.DataFrame(
        [
            {
                "start_sample": start,
                "end_sample": end,
                "duration_sec": (end - start) / comp_sr,
            }
            for start, end in intervals
        ]
    )

    stft = librosa.stft(full_y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    n_frames = magnitude.shape[1]
    frame_mask = _intervals_to_frame_mask(
        intervals=intervals,
        hop_length=HOP_LENGTH,
        n_frames=n_frames,
    )
    mag_vals = magnitude[:, frame_mask].flatten()

    fig_overlay = _plot_full_spectrogram_with_overlay(
        y=full_y,
        sr=full_sr,
        intervals=intervals,
    )
    image_overlay = fig_to_numpy_rgb(fig_overlay)

    fig_component = _plot_component_spectrogram(
        y=comp_y,
        sr=comp_sr,
    )
    image_component = fig_to_numpy_rgb(fig_component)

    fig_hist = _plot_histogram(mag_vals)
    image_hist = fig_to_numpy_rgb(fig_hist)

    if mag_vals.size == 0:
        summary = "No active intervals detected or no frames selected."
    else:
        median_mag = float(np.median(mag_vals))
        summary = (
            f"Intervals: {len(intervals)} | Frames used: {int(frame_mask.sum())} | "
            f"Median magnitude: {median_mag:.4e}"
        )

    analysis_state = {
        "full_y": full_y,
        "full_sr": full_sr,
        "stft": stft,
        "magnitude": magnitude,
        "frame_mask": frame_mask,
        "median_mag": float(np.median(mag_vals)) if mag_vals.size else 0.0,
        "interval_count": len(intervals),
    }

    return (
        image_overlay,
        image_component,
        image_hist,
        intervals_table,
        summary,
        analysis_state,
        None,
        None,
        None,
        "Run mask extraction to generate audio outputs.",
    )


def _apply_median_mask(state: dict | None, median_tolerance: float):
    if not state:
        return None, None, None, "Run analysis first."

    stft = state["stft"]
    magnitude = state["magnitude"]
    frame_mask = state["frame_mask"]
    full_y = state["full_y"]
    full_sr = state["full_sr"]
    median_mag = float(state["median_mag"])

    if median_mag <= 0 or not np.isfinite(median_mag):
        return None, None, None, "Median magnitude not valid for masking."

    tol = max(0.0, float(median_tolerance))
    band = median_mag * tol
    mask = np.abs(magnitude - median_mag) <= band
    mask[:, ~frame_mask] = False
    stft_masked = stft * mask
    y_est = librosa.istft(stft_masked, hop_length=HOP_LENGTH, length=len(full_y))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, y_est, full_sr)
        extracted_audio = tmp.name

    y_residual = (full_y - y_est).astype(np.float32)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, y_residual, full_sr)
        residual_audio = tmp.name

    fig_masked = _plot_masked_spectrogram(stft_masked, sr=full_sr)
    masked_image = fig_to_numpy_rgb(fig_masked)

    status = f"Masked frames only | median={median_mag:.4e} | tolerance={tol:.3f}"
    return masked_image, extracted_audio, residual_audio, status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Component Activity Analysis") as demo:
        gr.Markdown(
            "# Component Activity Analysis\n"
            "Find non-silent intervals in a component stem (dBFS threshold), then analyze "
            "power distribution on the full song spectrogram using those intervals."
        )

        with gr.Row():
            with gr.Column(scale=1):
                full_song = gr.Audio(
                    label="Full Song",
                    type="filepath",
                    sources=["upload"],
                )
                component = gr.Audio(
                    label="Component Stem",
                    type="filepath",
                    sources=["upload"],
                )
                silence_db = gr.Slider(
                    -80.0,
                    0.0,
                    value=-40.0,
                    step=1.0,
                    label="Silence Threshold (dBFS)",
                )
                median_tol = gr.Slider(
                    0.0,
                    1.0,
                    value=0.1,
                    step=0.01,
                    label="Median Tolerance (relative)",
                )
                run_btn = gr.Button("Run Analysis", variant="primary")
                mask_btn = gr.Button("Apply Median Mask")

            with gr.Column(scale=2):
                overlay_out = gr.Image(
                    label="Full Song STFT with Selected Frames", type="numpy"
                )
                component_spec_out = gr.Image(
                    label="Component Stem Spectrogram", type="numpy"
                )
                hist_out = gr.Image(
                    label="Magnitude Density (Selected Frames)", type="numpy"
                )
                intervals_out = gr.Dataframe(
                    label="Non-Silent Intervals (Sample Indices)"
                )
                summary_out = gr.Textbox(label="Summary")
                masked_spec_out = gr.Image(label="Masked Spectrogram", type="numpy")
                extracted_audio = gr.Audio(
                    label="Extracted Stem (Median Mask)", type="filepath"
                )
                residual_audio = gr.Audio(label="Residual Track", type="filepath")
                extraction_status = gr.Textbox(label="Mask Extraction Status")

        analysis_state = gr.State(value=None)

        run_btn.click(
            fn=_analyze_component,
            inputs=[full_song, component, silence_db],
            outputs=[
                overlay_out,
                component_spec_out,
                hist_out,
                intervals_out,
                summary_out,
                analysis_state,
                masked_spec_out,
                extracted_audio,
                residual_audio,
                extraction_status,
            ],
        )

        mask_btn.click(
            fn=_apply_median_mask,
            inputs=[analysis_state, median_tol],
            outputs=[
                masked_spec_out,
                extracted_audio,
                residual_audio,
                extraction_status,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

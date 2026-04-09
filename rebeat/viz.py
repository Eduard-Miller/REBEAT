from __future__ import annotations

import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from rebeat.analysis import DiagonalMatch, PatternGroup


def _pattern_color(pattern_id: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    r, g, b, _ = cmap(pattern_id % 20)
    return (float(r), float(g), float(b), 0.28)


def plot_spectrogram_with_patterns(
    y: np.ndarray,
    sr: int,
    patterns: tuple[PatternGroup, ...],
    hop_length: int = 512,
    n_fft: int = 2048,
) -> plt.Figure:
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("STFT Spectrogram with Repeating Pattern Highlights")

    legend_handles = []
    legend_labels = []
    for pattern in patterns:
        color = _pattern_color(pattern.pattern_id)
        for seg in pattern.segments:
            ax.axvspan(seg.start_sec, seg.end_sec, color=color)
        handle = plt.Rectangle((0, 0), 1, 1, color=color)
        legend_handles.append(handle)
        legend_labels.append(f"Pattern {pattern.pattern_id:02d}")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.8,
            ncol=2,
            fontsize=9,
        )
    return fig


def fig_to_numpy_rgb(fig: plt.Figure) -> np.ndarray:
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image = plt.imread(buffer)
    plt.close(fig)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return (image * 255).astype(np.uint8)


def plot_similarity_with_diagonals(
    similarity: np.ndarray,
    diagonal_matches: tuple[DiagonalMatch, ...],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    img = ax.imshow(
        similarity,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=-0.1,
        vmax=1.0,
    )
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03, label="Cosine Similarity")
    ax.set_title("Self-Similarity Matrix with Detected Diagonals")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Frame Index")

    for match in diagonal_matches:
        ax.plot(
            [match.j0, match.j0 + match.length],
            [match.i0, match.i0 + match.length],
            color="#ff3b30",
            linewidth=1.5,
            alpha=0.95,
        )
    return fig


def plot_onset_autocorrelation(
    onset_autocorr: np.ndarray | None,
    sr: int,
    hop_length: int,
    best_lag: int | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.2), constrained_layout=True)
    if onset_autocorr is None or onset_autocorr.size == 0:
        ax.text(
            0.5,
            0.5,
            "No onset autocorrelation available",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        return fig

    lags = np.arange(onset_autocorr.size)
    times = librosa.frames_to_time(lags, sr=sr, hop_length=hop_length)
    ax.plot(times, onset_autocorr, color="#005f73", linewidth=1.6)
    ax.set_title("Onset Autocorrelation (Loop Detection Signal)")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Autocorr")
    ax.grid(alpha=0.2)

    if best_lag is not None and 0 <= best_lag < onset_autocorr.size:
        best_t = float(librosa.frames_to_time(best_lag, sr=sr, hop_length=hop_length))
        best_v = float(onset_autocorr[best_lag])
        ax.axvline(best_t, color="#d00000", linestyle="--", linewidth=1.4)
        ax.scatter([best_t], [best_v], color="#d00000", s=26, zorder=3)
        ax.text(
            best_t,
            best_v,
            f" lag={best_t:.2f}s",
            color="#d00000",
            fontsize=9,
            va="bottom",
        )
    return fig


def plot_analyzed_window_spectrogram(
    y: np.ndarray,
    sr: int,
    analyzed_duration_sec: float,
    loop_optimization_applied: bool,
    patterns: tuple[PatternGroup, ...],
    hop_length: int = 512,
    n_fft: int = 2048,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    if not loop_optimization_applied:
        ax.text(
            0.5,
            0.5,
            "Loop optimization not applied\n(full track was analyzed)",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        return fig

    analyzed_samples = int(round(analyzed_duration_sec * sr))
    if analyzed_samples <= 0:
        ax.text(
            0.5,
            0.5,
            "No analyzed loop segment available",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_axis_off()
        return fig

    segment = y[: min(len(y), analyzed_samples)]
    stft = librosa.stft(y=segment, n_fft=n_fft, hop_length=hop_length)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Analyzed Loop Segment Spectrogram")

    legend_handles = []
    legend_labels = []
    for pattern in patterns:
        color = _pattern_color(pattern.pattern_id)
        has_visible = False
        for seg in pattern.segments:
            if seg.end_sec <= 0 or seg.start_sec >= analyzed_duration_sec:
                continue
            start = max(0.0, seg.start_sec)
            end = min(analyzed_duration_sec, seg.end_sec)
            if end <= start:
                continue
            ax.axvspan(start, end, color=color)
            has_visible = True
        if has_visible:
            handle = plt.Rectangle((0, 0), 1, 1, color=color)
            legend_handles.append(handle)
            legend_labels.append(f"Pattern {pattern.pattern_id:02d}")

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.8,
            ncol=2,
            fontsize=8,
        )
    return fig

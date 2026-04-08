from __future__ import annotations

import argparse
from pathlib import Path

from rebeat.analysis import analyze_repeats, patterns_to_table
from rebeat.viz import (
    plot_onset_autocorrelation,
    plot_similarity_with_diagonals,
    plot_spectrogram_with_patterns,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect repeating music patterns and visualize them on a spectrogram."
    )
    parser.add_argument("audio", type=Path, help="Path to input audio file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rebeat_spectrogram.png"),
        help="Output path for the spectrogram image.",
    )
    parser.add_argument(
        "--similarity-output",
        type=Path,
        default=Path("rebeat_similarity.png"),
        help="Output path for the similarity matrix image.",
    )
    parser.add_argument(
        "--autocorr-output",
        type=Path,
        default=Path("rebeat_onset_autocorr.png"),
        help="Output path for onset autocorrelation image.",
    )
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length.")
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT FFT size.")
    parser.add_argument(
        "--min-repeat-sec",
        type=float,
        default=2.5,
        help="Minimum duration for a repeating segment.",
    )
    parser.add_argument(
        "--min-lag-sec",
        type=float,
        default=2.0,
        help="Minimum temporal separation between repeated segments.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.72,
        help="Similarity threshold in [0, 1].",
    )
    parser.add_argument(
        "--max-patterns",
        type=int,
        default=12,
        help="Maximum number of detected pattern groups to keep.",
    )
    parser.add_argument(
        "--no-mfcc",
        action="store_true",
        help="Disable MFCC features and use chroma only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis = analyze_repeats(
        audio_path=args.audio,
        sr=args.sr,
        hop_length=args.hop_length,
        min_repeat_sec=args.min_repeat_sec,
        min_lag_sec=args.min_lag_sec,
        similarity_threshold=args.similarity_threshold,
        max_patterns=args.max_patterns,
        use_mfcc=not args.no_mfcc,
    )

    fig = plot_spectrogram_with_patterns(
        y=analysis.y,
        sr=analysis.sr,
        patterns=analysis.patterns,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")

    fig_similarity = plot_similarity_with_diagonals(
        similarity=analysis.similarity,
        diagonal_matches=analysis.diagonal_matches,
    )
    args.similarity_output.parent.mkdir(parents=True, exist_ok=True)
    fig_similarity.savefig(args.similarity_output, dpi=160, bbox_inches="tight")

    fig_autocorr = plot_onset_autocorrelation(
        onset_autocorr=analysis.onset_autocorr,
        sr=analysis.sr,
        hop_length=analysis.hop_length,
        best_lag=analysis.onset_autocorr_best_lag,
    )
    args.autocorr_output.parent.mkdir(parents=True, exist_ok=True)
    fig_autocorr.savefig(args.autocorr_output, dpi=160, bbox_inches="tight")

    table = patterns_to_table(analysis.patterns)
    print(f"Saved spectrogram: {args.output}")
    print(f"Saved similarity matrix: {args.similarity_output}")
    print(f"Saved onset autocorrelation: {args.autocorr_output}")
    loop = analysis.loop_optimization
    if loop.applied:
        print(
            "Loop optimization: applied "
            f"(loop {loop.loop_duration_sec:.2f}s, "
            f"estimated count {loop.loop_count}, "
            f"analyzed {loop.analyzed_duration_sec:.2f}s)."
        )
    else:
        print("Loop optimization: not applied (full track analyzed).")
    if not table:
        print("No repeating patterns detected with current settings.")
        return
    print("Detected repeating patterns:")
    for row in table:
        print(
            f"- pattern_{row['pattern_id']:02d}: "
            f"{row['start_sec']:.2f}s to {row['end_sec']:.2f}s "
            f"(duration {row['duration_sec']:.2f}s)"
        )


if __name__ == "__main__":
    main()

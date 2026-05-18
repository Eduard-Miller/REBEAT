from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class StemScore:
    track_id: str
    stem_id: str
    file_path: str
    duration_sec: float
    score: float
    p95: float
    p99: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score BabySlakh stems by self-similarity (higher means more repetitive)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/home/bordduk69/Nextcloud/NTNU/10. Vår2026/Master/Code/babyslakh_16k"
        ),
        help="Path to BabySlakh root with Track*/stems/*.wav.",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument(
        "--hop-length",
        type=int,
        default=2048,
        help="Hop length for frame features (larger is faster).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Keep every Nth frame before similarity (larger is faster).",
    )
    parser.add_argument(
        "--min-lag-sec",
        type=float,
        default=1.0,
        help="Ignore matches closer than this time distance.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many top stems to print.",
    )
    parser.add_argument(
        "--silence-top-db",
        type=float,
        default=40.0,
        help="Silence threshold (dB) for non-silent segmentation.",
    )
    parser.add_argument(
        "--min-active-sec",
        type=float,
        default=1.0,
        help="Minimum non-silent duration required for scoring.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for all stem scores.",
    )
    return parser.parse_args()


def extract_features(
    y: np.ndarray, sr: int, hop_length: int, frame_step: int
) -> np.ndarray:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    feats = np.vstack([chroma, mfcc]).astype(np.float32)

    if frame_step > 1:
        feats = feats[:, ::frame_step]

    mean = np.mean(feats, axis=1, keepdims=True)
    std = np.std(feats, axis=1, keepdims=True) + 1e-8
    return (feats - mean) / std


def score_stem(
    stem_path: Path,
    sr: int,
    hop_length: int,
    frame_step: int,
    min_lag_sec: float,
    silence_top_db: float,
    min_active_sec: float,
) -> StemScore:
    y, loaded_sr = librosa.load(stem_path.as_posix(), sr=sr, mono=True)
    duration_sec = float(len(y) / loaded_sr)

    intervals = librosa.effects.split(y, top_db=silence_top_db)
    if intervals.size == 0:
        active_y = np.array([], dtype=np.float32)
    else:
        chunks = [y[start:end] for start, end in intervals if end > start]
        active_y = (
            np.concatenate(chunks).astype(np.float32)
            if chunks
            else np.array([], dtype=np.float32)
        )

    if float(len(active_y) / loaded_sr) < min_active_sec:
        return StemScore(
            track_id=stem_path.parent.parent.name,
            stem_id=stem_path.stem,
            file_path=stem_path.as_posix(),
            duration_sec=duration_sec,
            score=0.0,
            p95=0.0,
            p99=0.0,
        )

    feats = extract_features(
        y=active_y, sr=loaded_sr, hop_length=hop_length, frame_step=frame_step
    )
    n_frames = feats.shape[1]
    if n_frames < 8:
        return StemScore(
            track_id=stem_path.parent.parent.name,
            stem_id=stem_path.stem,
            file_path=stem_path.as_posix(),
            duration_sec=duration_sec,
            score=0.0,
            p95=0.0,
            p99=0.0,
        )

    sim = cosine_similarity(feats.T).astype(np.float32)

    min_lag_frames = max(
        1,
        int(np.ceil((min_lag_sec * loaded_sr) / (hop_length * max(1, frame_step)))),
    )
    i_idx, j_idx = np.triu_indices(n_frames, k=min_lag_frames)
    vals = sim[i_idx, j_idx]
    if vals.size == 0:
        p95 = 0.0
        p99 = 0.0
        score = 0.0
    else:
        p95 = float(np.quantile(vals, 0.95))
        p99 = float(np.quantile(vals, 0.99))
        score = 0.6 * p95 + 0.4 * p99

    return StemScore(
        track_id=stem_path.parent.parent.name,
        stem_id=stem_path.stem,
        file_path=stem_path.as_posix(),
        duration_sec=duration_sec,
        score=score,
        p95=p95,
        p99=p99,
    )


def print_rankings(scores: list[StemScore], top_k: int) -> None:
    if not scores:
        print("No stems found.")
        return

    print("Top stems by self-similarity score")
    print("rank\ttrack\tstem\tscore\tp95\tp99\tduration_s")
    for rank, item in enumerate(scores[:top_k], start=1):
        print(
            f"{rank}\t{item.track_id}\t{item.stem_id}\t{item.score:.4f}\t"
            f"{item.p95:.4f}\t{item.p99:.4f}\t{item.duration_sec:.2f}"
        )

    print("\nBest stem per track")
    print("track\tstem\tscore\tp95\tp99\tduration_s")
    best_by_track: dict[str, StemScore] = {}
    for item in scores:
        cur = best_by_track.get(item.track_id)
        if cur is None or item.score > cur.score:
            best_by_track[item.track_id] = item

    for track_id in sorted(best_by_track):
        item = best_by_track[track_id]
        print(
            f"{track_id}\t{item.stem_id}\t{item.score:.4f}\t{item.p95:.4f}\t"
            f"{item.p99:.4f}\t{item.duration_sec:.2f}"
        )


def write_csv(csv_path: Path, scores: list[StemScore]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "stem_id",
                "score",
                "p95",
                "p99",
                "duration_sec",
                "file_path",
            ],
        )
        writer.writeheader()
        for item in scores:
            writer.writerow(
                {
                    "track_id": item.track_id,
                    "stem_id": item.stem_id,
                    "score": f"{item.score:.6f}",
                    "p95": f"{item.p95:.6f}",
                    "p99": f"{item.p99:.6f}",
                    "duration_sec": f"{item.duration_sec:.6f}",
                    "file_path": item.file_path,
                }
            )


def main() -> None:
    args = parse_args()
    stem_paths = sorted(
        p
        for p in args.dataset.glob("Track*/stems/*.wav")
        if not p.name.startswith(".") and not p.name.startswith("._")
    )
    if not stem_paths:
        print(f"No stems found at: {args.dataset}")
        return

    print(f"Found {len(stem_paths)} stems. Scoring...")
    scores: list[StemScore] = []

    for idx, stem_path in enumerate(stem_paths, start=1):
        try:
            item = score_stem(
                stem_path=stem_path,
                sr=args.sr,
                hop_length=args.hop_length,
                frame_step=args.frame_step,
                min_lag_sec=args.min_lag_sec,
                silence_top_db=args.silence_top_db,
                min_active_sec=args.min_active_sec,
            )
            scores.append(item)
            print(
                f"[{idx:03d}/{len(stem_paths):03d}] {item.track_id}/{item.stem_id} "
                f"score={item.score:.4f}"
            )
        except Exception as exc:
            print(f"[{idx:03d}/{len(stem_paths):03d}] ERROR {stem_path}: {exc}")

    scores.sort(key=lambda x: x.score, reverse=True)
    print()
    print_rankings(scores=scores, top_k=max(1, args.top_k))

    if args.csv is not None:
        write_csv(args.csv, scores)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()

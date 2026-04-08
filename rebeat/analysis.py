from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import label
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class PatternGroup:
    pattern_id: int
    segments: tuple[Segment, ...]
    score: float


@dataclass(frozen=True)
class DiagonalMatch:
    i0: int
    j0: int
    length: int
    score: float


@dataclass(frozen=True)
class LoopOptimizationInfo:
    applied: bool
    loop_duration_sec: float
    loop_count: int
    analyzed_duration_sec: float


@dataclass(frozen=True)
class AnalysisResult:
    y: np.ndarray
    sr: int
    hop_length: int
    similarity: np.ndarray
    diagonal_matches: tuple[DiagonalMatch, ...]
    patterns: tuple[PatternGroup, ...]
    loop_optimization: LoopOptimizationInfo
    onset_autocorr: np.ndarray | None
    onset_autocorr_best_lag: int | None


def _extract_features(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    use_mfcc: bool,
) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    features = [chroma]
    if use_mfcc:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        features.append(mfcc)
    stacked = np.vstack(features).astype(np.float32)
    mean = np.mean(stacked, axis=1, keepdims=True)
    std = np.std(stacked, axis=1, keepdims=True) + 1e-8
    return (stacked - mean) / std


def _frame_to_sec(frame_idx: int, sr: int, hop_length: int) -> float:
    return float(librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length))


def _compute_similarity(features: np.ndarray) -> np.ndarray:
    return cosine_similarity(features.T).astype(np.float32)


def _extract_pairs_from_similarity(
    similarity: np.ndarray,
    threshold: float,
    min_lag_frames: int,
    min_len_frames: int,
) -> list[DiagonalMatch]:
    n_frames = similarity.shape[0]
    sim = similarity.copy()
    for k in range(-min_lag_frames, min_lag_frames + 1):
        i = np.arange(max(0, -k), min(n_frames, n_frames - k))
        j = i + k
        sim[i, j] = 0.0

    binary = sim >= threshold
    labeled, n_components = label(binary)
    pairs: list[DiagonalMatch] = []

    for comp_id in range(1, n_components + 1):
        coords = np.argwhere(labeled == comp_id)
        if coords.size == 0:
            continue

        diag_map: dict[int, list[tuple[int, int]]] = {}
        for i, j in coords:
            diag_map.setdefault(int(j - i), []).append((int(i), int(j)))

        for points in diag_map.values():
            points.sort()
            run: list[tuple[int, int]] = []
            for p in points:
                if not run:
                    run = [p]
                    continue
                prev_i, prev_j = run[-1]
                if p[0] == prev_i + 1 and p[1] == prev_j + 1:
                    run.append(p)
                else:
                    if len(run) >= min_len_frames:
                        i0, j0 = run[0]
                        score = float(np.mean([sim[a, b] for a, b in run]))
                        pairs.append(
                            DiagonalMatch(i0=i0, j0=j0, length=len(run), score=score)
                        )
                    run = [p]

            if len(run) >= min_len_frames:
                i0, j0 = run[0]
                score = float(np.mean([sim[a, b] for a, b in run]))
                pairs.append(DiagonalMatch(i0=i0, j0=j0, length=len(run), score=score))

    return pairs


def _interval_overlap(a: Segment, b: Segment) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _segment_union(a: Segment, b: Segment) -> Segment:
    return Segment(
        start_sec=min(a.start_sec, b.start_sec), end_sec=max(a.end_sec, b.end_sec)
    )


def _dedupe_segments(
    segments: list[Segment], overlap_ratio: float = 0.5
) -> list[Segment]:
    merged: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start_sec, s.end_sec)):
        matched = False
        for idx, cur in enumerate(merged):
            ov = _interval_overlap(seg, cur)
            min_len = min(seg.end_sec - seg.start_sec, cur.end_sec - cur.start_sec)
            if min_len <= 0:
                continue
            if ov / min_len >= overlap_ratio:
                merged[idx] = _segment_union(seg, cur)
                matched = True
                break
        if not matched:
            merged.append(seg)
    return merged


def _build_patterns(
    pairs: list[DiagonalMatch],
    sr: int,
    hop_length: int,
    max_patterns: int,
) -> tuple[PatternGroup, ...]:
    pattern_candidates: list[tuple[list[Segment], float]] = []

    for pair in pairs:
        seg_a = Segment(
            start_sec=_frame_to_sec(pair.i0, sr, hop_length),
            end_sec=_frame_to_sec(pair.i0 + pair.length, sr, hop_length),
        )
        seg_b = Segment(
            start_sec=_frame_to_sec(pair.j0, sr, hop_length),
            end_sec=_frame_to_sec(pair.j0 + pair.length, sr, hop_length),
        )
        pattern_candidates.append(([seg_a, seg_b], pair.score))

    pattern_candidates.sort(key=lambda x: x[1], reverse=True)
    groups: list[PatternGroup] = []

    for segments, score in pattern_candidates:
        merged_in = False
        for idx, group in enumerate(groups):
            g_segments = list(group.segments)
            if any(_interval_overlap(s, gs) > 0 for s in segments for gs in g_segments):
                all_segments = _dedupe_segments(g_segments + segments)
                groups[idx] = PatternGroup(
                    pattern_id=group.pattern_id,
                    segments=tuple(all_segments),
                    score=max(group.score, score),
                )
                merged_in = True
                break
        if not merged_in:
            groups.append(
                PatternGroup(
                    pattern_id=len(groups),
                    segments=tuple(_dedupe_segments(segments)),
                    score=score,
                )
            )
        if len(groups) >= max_patterns:
            break

    filtered = [g for g in groups if len(g.segments) >= 2]
    reindexed = [
        PatternGroup(pattern_id=i, segments=g.segments, score=g.score)
        for i, g in enumerate(filtered)
    ]
    return tuple(reindexed)


def _estimate_loop_frames(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    min_loop_sec: float,
    max_loop_ratio: float,
    onset_threshold: float,
) -> tuple[int, int, np.ndarray | None, int | None]:
    duration_sec = float(len(y) / sr)
    if duration_sec < max(20.0, min_loop_sec * 2.0):
        return 0, 1, None, None

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset_env.size < 8:
        return 0, 1, None, None

    onset_env = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)
    corr = librosa.autocorrelate(onset_env, max_size=onset_env.size)
    corr = corr / (corr[0] + 1e-8)

    min_lag = max(1, int(np.ceil(min_loop_sec * sr / hop_length)))
    max_lag = max(min_lag + 1, int(np.floor(max_loop_ratio * onset_env.size)))
    if max_lag <= min_lag:
        return 0, 1, corr, None

    window = corr[min_lag:max_lag]
    if window.size == 0:
        return 0, 1, corr, None

    best_rel = int(np.argmax(window))
    best_score = float(window[best_rel])
    best_lag = min_lag + best_rel
    if best_score < onset_threshold:
        return 0, 1, corr, best_lag

    loop_frames = best_lag
    estimated_count = int(np.floor(onset_env.size / loop_frames))
    if estimated_count < 2:
        return 0, 1, corr, best_lag
    return loop_frames, estimated_count, corr, best_lag


def _lift_patterns_to_full_track(
    base_patterns: tuple[PatternGroup, ...],
    loop_duration_sec: float,
    total_duration_sec: float,
) -> tuple[PatternGroup, ...]:
    if loop_duration_sec <= 0:
        return base_patterns

    lifted: list[PatternGroup] = []
    for group in base_patterns:
        segments: list[Segment] = []
        k = 0
        while True:
            offset = k * loop_duration_sec
            if offset >= total_duration_sec:
                break
            for seg in group.segments:
                start = seg.start_sec + offset
                end = seg.end_sec + offset
                if start >= total_duration_sec:
                    continue
                segments.append(
                    Segment(start_sec=start, end_sec=min(end, total_duration_sec))
                )
            k += 1

        deduped = _dedupe_segments(segments, overlap_ratio=0.65)
        if len(deduped) >= 2:
            lifted.append(
                PatternGroup(
                    pattern_id=len(lifted),
                    segments=tuple(deduped),
                    score=group.score,
                )
            )
    return tuple(lifted)


def analyze_repeats(
    audio_path: str | Path,
    sr: int = 22050,
    hop_length: int = 512,
    min_repeat_sec: float = 2.5,
    min_lag_sec: float = 2.0,
    similarity_threshold: float = 0.72,
    max_patterns: int = 12,
    use_mfcc: bool = True,
    optimize_for_loops: bool = True,
    min_loop_sec: float = 20.0,
    max_loop_ratio: float = 0.6,
    loop_onset_threshold: float = 0.28,
) -> AnalysisResult:
    y, loaded_sr = librosa.load(path=str(audio_path), sr=sr, mono=True)

    total_duration_sec = float(len(y) / loaded_sr)
    analyzed_y = y
    loop_duration_sec = 0.0
    loop_count = 1
    loop_applied = False
    onset_autocorr: np.ndarray | None = None
    onset_autocorr_best_lag: int | None = None

    if optimize_for_loops:
        loop_frames, estimated_count, onset_autocorr, onset_autocorr_best_lag = (
            _estimate_loop_frames(
                y=y,
                sr=loaded_sr,
                hop_length=hop_length,
                min_loop_sec=min_loop_sec,
                max_loop_ratio=max_loop_ratio,
                onset_threshold=loop_onset_threshold,
            )
        )
        if loop_frames > 0 and estimated_count >= 2:
            loop_duration_sec = _frame_to_sec(loop_frames, loaded_sr, hop_length)
            loop_samples = int(round(loop_duration_sec * loaded_sr))
            if 0 < loop_samples < len(y):
                analyzed_y = y[:loop_samples]
                loop_count = int(np.floor(total_duration_sec / loop_duration_sec))
                loop_applied = loop_count >= 2

    features = _extract_features(
        y=analyzed_y, sr=loaded_sr, hop_length=hop_length, use_mfcc=use_mfcc
    )
    similarity = _compute_similarity(features)

    min_len_frames = max(1, int(np.ceil(min_repeat_sec * loaded_sr / hop_length)))
    min_lag_frames = max(1, int(np.ceil(min_lag_sec * loaded_sr / hop_length)))
    pairs = _extract_pairs_from_similarity(
        similarity=similarity,
        threshold=similarity_threshold,
        min_lag_frames=min_lag_frames,
        min_len_frames=min_len_frames,
    )

    patterns = _build_patterns(
        pairs=pairs,
        sr=loaded_sr,
        hop_length=hop_length,
        max_patterns=max_patterns,
    )

    if loop_applied:
        patterns = _lift_patterns_to_full_track(
            base_patterns=patterns,
            loop_duration_sec=loop_duration_sec,
            total_duration_sec=total_duration_sec,
        )

    loop_info = LoopOptimizationInfo(
        applied=loop_applied,
        loop_duration_sec=loop_duration_sec,
        loop_count=loop_count,
        analyzed_duration_sec=float(len(analyzed_y) / loaded_sr),
    )

    return AnalysisResult(
        y=y,
        sr=loaded_sr,
        hop_length=hop_length,
        similarity=similarity,
        diagonal_matches=tuple(pairs),
        patterns=patterns,
        loop_optimization=loop_info,
        onset_autocorr=onset_autocorr,
        onset_autocorr_best_lag=onset_autocorr_best_lag,
    )


def patterns_to_table(
    patterns: tuple[PatternGroup, ...],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for pattern in patterns:
        for seg in pattern.segments:
            rows.append(
                {
                    "pattern_id": pattern.pattern_id,
                    "start_sec": round(seg.start_sec, 3),
                    "end_sec": round(seg.end_sec, 3),
                    "duration_sec": round(seg.end_sec - seg.start_sec, 3),
                    "score": round(pattern.score, 4),
                }
            )
    rows.sort(key=lambda x: (int(x["pattern_id"]), float(x["start_sec"])))
    return rows

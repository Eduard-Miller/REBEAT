import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from sortedcontainers import SortedSet

def calc_STFT(path: str, sr, duration, hop_length, win_length, window_type) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, duration=duration)
    stft_result = librosa.stft(y, hop_length=hop_length, win_length=win_length, window=window_type)
    return stft_result

def calc_magnitude(stft: np.ndarray) -> np.ndarray:
    return np.abs(stft)

def calc_iSTFT(data: np.ndarray, hop_length, win_length, window_type) -> np.ndarray:
    return librosa.istft(data, hop_length=hop_length, win_length=win_length, window=window_type)

def calc_sim_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculates the similarity matrix S from the magnitude spectrogram V.
    V shape expected: (num_frequency_bins, num_time_frames)
    Output S shape: (num_time_frames, num_time_frames)
    """
    return cosine_similarity(data.T).astype(np.float32)

def get_top_k_similar_frames(sim_matrix: np.ndarray, frame_index: int, k: int) -> SortedSet:
    frame_similarities = sim_matrix[frame_index]
    top_k_indices = np.argsort(frame_similarities)[::-1][1:k+1]  # Exclude self-similarity
    return SortedSet(top_k_indices)

def median_filter(data: np.ndarray, frames: SortedSet) -> np.ndarray:
    # Apply width wise median filtering (per frequency bin) to the specified frames and 0 out the rest
    filtered_data = np.zeros_like(data)
    median = np.median(data[list(frames)], axis=0)
    for frame in frames:
        filtered_data[frame] = median
    return filtered_data

def calc_mask(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    # Calculate the mask for the given V and W matrices
    if W.shape != V.shape:
        raise ValueError("W and V must have the same shape for mask calculation.")
    
    W_marked = np.minimum(W, V) # represents repeating model for the "background" in REPET
    M = W_marked/(V + 1e-10) # Avoid division by zero, values in M should be between 0 and 1
    
    return M

def apply_mask(data: np.ndarray, M: np.ndarray) -> np.ndarray:
    return data * M

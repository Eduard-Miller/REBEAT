import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

class MusicProcessingProvider:
    path = ""
    sr = None
    stft = None
    magnitude = None
    sim_matrix = None
    
    
    def __init__(self, path: str, duration: float, nfft: int, hop_length: int, win_length: int, window_type: str):
        self.path = path
        self.stft = self.calc_STFT(duration, nfft, hop_length, win_length, window_type)
        self.magnitude = self.calc_magnitude()
        self.sim_matrix = self.calc_sim_matrix()

    def calc_STFT(self, duration: float, nfft: int, hop_length: int, win_length: int, window_type: str) -> np.ndarray:
        y, self.sr = librosa.load(self.path, sr=None, duration=duration)
        stft_result = librosa.stft(y, n_fft=nfft, hop_length=hop_length, win_length=win_length, window=window_type)
        return stft_result

    def calc_magnitude(self) -> np.ndarray:
        return np.abs(self.stft)

    def calc_iSTFT(self, data: np.ndarray, hop_length: int, win_length: int, window_type: str) -> np.ndarray:
        return librosa.istft(data, hop_length=hop_length, win_length=win_length, window=window_type)

    def calc_sim_matrix(self) -> np.ndarray:
        """
        Calculates the similarity matrix S from the magnitude spectrogram V.
        V shape expected: (num_frequency_bins, num_time_frames)
        Output S shape: (num_time_frames, num_time_frames)
        """
        return cosine_similarity(self.magnitude.T).astype(np.float32)

    def get_top_k_similar_frames(self, frame_index: int, k: int) -> np.ndarray:
        frame_similarities = self.sim_matrix[frame_index]
        top_k_indices = np.argsort(frame_similarities)[::-1][1:k+1]  # Exclude self-similarity
        return top_k_indices

    def median_filter(self, frames: np.ndarray) -> np.ndarray:
        # Apply width wise median filtering (per frequency bin) to the specified frames and 0 out the rest creating the repeating model W
        filtered_data = np.zeros_like(self.magnitude)
        median = np.median(self.magnitude[:, frames], axis=1)
        for frame in frames:
            filtered_data[:, frame] = median
        return filtered_data

    def calc_median_frame(self, frames: np.ndarray) -> np.ndarray:
        return np.median(self.magnitude[:, frames], axis=1)

    def calc_mask(self, W: np.ndarray) -> np.ndarray:
        V = self.magnitude
        # Calculate the mask for the given V and W matrices
        if W.shape != V.shape:
            raise ValueError("W and V must have the same shape for mask calculation.")
        
        W_marked = np.minimum(W, V) # represents repeating model for the "background" in REPET
        M = W_marked/(V + 1e-10) # Avoid division by zero, values in M should be between 0 and 1
        
        return M

    def apply_mask(self, data: np.ndarray, M: np.ndarray) -> np.ndarray:
        masked_data = data * M
    
        # Add an extremely tiny amount of random phase/energy floor (dither) 
        # so that silent frames aren't absolute zero, preventing singular matrices in museval
        floor = 1e-8 * np.random.randn(*masked_data.shape) + 1e-8 * 1j * np.random.randn(*masked_data.shape)

        # Only apply the floor where the mask has zeroed things out
        masked_data = np.where(M == 0, floor, masked_data)

        return masked_data
"""
Audio Preprocessor Module
High-quality audio preprocessing for game music MML generation
"""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple


class AudioPreprocessor:
    """High-quality audio preprocessing for game music MML generation"""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize audio preprocessor

        Args:
            sample_rate: Target sample rate (44100 Hz for high quality)
        """
        self.sample_rate = sample_rate

    def load_and_preprocess(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If audio loading fails
        """
        # Check if file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            # Normalize amplitude to [-1, 1]
            audio = librosa.util.normalize(audio)

            # Trim silence from beginning and end (top_db=20)
            audio, _ = librosa.effects.trim(audio, top_db=20)

            return audio, self.sample_rate

        except Exception as e:
            raise RuntimeError(f"Failed to load and preprocess audio: {e}")

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate

        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1]

        Args:
            audio: Audio array

        Returns:
            Normalized audio array
        """
        return librosa.util.normalize(audio)

    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Remove silence from beginning and end

        Args:
            audio: Audio array
            top_db: Threshold in dB below reference to consider as silence

        Returns:
            Trimmed audio array
        """
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed

    def segment_audio(self, audio: np.ndarray, segment_length: float,
                     sr: int) -> list[np.ndarray]:
        """
        Split audio into segments of specified length

        Args:
            audio: Audio array
            segment_length: Length of each segment in seconds
            sr: Sample rate

        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * sr)
        segments = []

        for start in range(0, len(audio), segment_samples):
            end = min(start + segment_samples, len(audio))
            segments.append(audio[start:end])

        return segments


def main():
    """Example usage of AudioPreprocessor"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_preprocessor.py <audio_file_path>")
        return

    audio_path = sys.argv[1]

    preprocessor = AudioPreprocessor(sample_rate=44100)

    print(f"Loading and preprocessing: {audio_path}")
    audio, sr = preprocessor.load_and_preprocess(audio_path)

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    print(f"Min amplitude: {audio.min():.3f}")
    print(f"Max amplitude: {audio.max():.3f}")


if __name__ == "__main__":
    main()

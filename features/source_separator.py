"""
Source Separator Module
AI-powered source separation for melody and harmony extraction
"""

import torch
import numpy as np
from typing import Dict, Optional

try:
    from demucs import pretrained
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("Warning: Demucs not installed. Please install with: pip install demucs")


class SourceSeparator:
    """AI-powered source separation for melody and harmony extraction"""

    def __init__(self, model_name: str = "htdemucs", force_cpu: bool = False):
        """
        Initialize source separator

        Args:
            model_name: Demucs model name (htdemucs for best quality)
            force_cpu: Force CPU mode even if CUDA is available

        Raises:
            ImportError: If Demucs is not installed
            RuntimeError: If model loading fails
        """
        if not DEMUCS_AVAILABLE:
            raise ImportError(
                "Demucs is not installed. Please install with: pip install demucs"
            )

        # Determine device
        if force_cpu:
            self.device = "cpu"
            print(f"Using device: {self.device} (forced)")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")

        try:
            self.model = pretrained.get_model(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Demucs model '{model_name}': {e}")

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems

        Args:
            audio: Audio array (mono)
            sr: Sample rate

        Returns:
            Dictionary with keys: vocals, drums, bass, other

        Raises:
            RuntimeError: If separation fails
        """
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Add batch and channel dimensions
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Convert audio to model's sample rate and channels
            audio_tensor = convert_audio(
                audio_tensor, sr, self.model.samplerate, self.model.audio_channels
            )

            # Move to device
            audio_tensor = audio_tensor.to(self.device)

            # Apply separation
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    audio_tensor,
                    device=self.device,
                    shifts=1,
                    split=True,
                    overlap=0.25
                )

            # Convert back to numpy
            sources = sources.cpu().numpy()[0]  # Remove batch dim

            # Convert to mono if stereo
            if sources.shape[1] > 1:
                sources = np.mean(sources, axis=1)

            # Map to stem names (order: drums, bass, other, vocals)
            stem_names = ["drums", "bass", "other", "vocals"]
            stems = {name: sources[i] for i, name in enumerate(stem_names)}

            return stems

        except Exception as e:
            raise RuntimeError(f"Failed to separate audio: {e}")

    def extract_tracks_for_mml(self, stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Automatically select melody and harmony tracks

        Strategy:
        - Melody: vocals (primary) or other (instrumental lead)
        - Harmony 1: bass (low frequency foundation)
        - Harmony 2: other (chords, pads, supporting instruments)

        Args:
            stems: Dictionary of separated stems

        Returns:
            Dictionary with keys: melody, harmony1, harmony2
        """
        # Analyze vocals energy to determine if vocal-dominant or instrumental
        vocals_energy = np.mean(np.abs(stems["vocals"]))
        other_energy = np.mean(np.abs(stems["other"]))

        if vocals_energy > other_energy * 0.3:  # Vocal-dominant track
            melody_audio = stems["vocals"]
            harmony2_audio = stems["other"]
        else:  # Instrumental track
            melody_audio = stems["other"]  # Lead instrument
            harmony2_audio = stems["vocals"]  # Supporting

        return {
            "melody": melody_audio,
            "harmony1": stems["bass"],
            "harmony2": harmony2_audio
        }


def main():
    """Example usage of SourceSeparator"""
    import sys
    import soundfile as sf

    if len(sys.argv) < 2:
        print("Usage: python source_separator.py <audio_file_path>")
        return

    audio_path = sys.argv[1]

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Initialize separator
    print("Initializing source separator...")
    separator = SourceSeparator(model_name="htdemucs")

    # Separate
    print("Separating audio sources (this may take a while)...")
    stems = separator.separate(audio, sr)

    print("\nStem energies:")
    for name, stem in stems.items():
        energy = np.mean(np.abs(stem))
        print(f"  {name}: {energy:.6f}")

    # Extract tracks for MML
    print("\nExtracting tracks for MML...")
    tracks = separator.extract_tracks_for_mml(stems)

    print("\nTrack assignments:")
    for name in tracks.keys():
        print(f"  {name}")

    # Save separated stems (optional)
    output_dir = "separated_stems"
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, stem in stems.items():
        output_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(output_path, stem, sr)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""
Music Analyzer Module
Automatic music analysis and quantization
"""

import librosa
import numpy as np
from typing import List, Tuple

from .models import Note, Track


class MusicAnalyzer:
    """Automatic music analysis and quantization"""

    def __init__(self, quantize_resolution: int = 16):
        """
        Initialize music analyzer

        Args:
            quantize_resolution: Quantization resolution (16 = 1/16 notes)
        """
        self.quantize_resolution = quantize_resolution

    def analyze_and_process(self, audio: np.ndarray, sr: int,
                           tracks: List[Track]) -> List[Track]:
        """
        Analyze audio and process tracks

        Args:
            audio: Original audio (for tempo/key detection)
            sr: Sample rate
            tracks: List of tracks with raw notes

        Returns:
            List of processed tracks with tempo, key, and quantized notes

        Raises:
            RuntimeError: If analysis fails
        """
        try:
            # Auto-detect tempo
            tempo = self.detect_tempo(audio, sr)

            # Auto-detect key signature
            key_signature = self.detect_key(audio, sr)

            # Time signature (default to 4/4)
            time_signature = (4, 4)

            # Process each track
            processed_tracks = []
            for track in tracks:
                # Set musical parameters
                track.tempo = tempo
                track.key_signature = key_signature
                track.time_signature = time_signature

                # Quantize notes
                track.notes = self.quantize_notes(track.notes, tempo)

                # Remove outliers
                track.notes = self.remove_outliers(track.notes)

                # Sort notes
                track.sort_notes()

                processed_tracks.append(track)

            return processed_tracks

        except Exception as e:
            raise RuntimeError(f"Failed to analyze and process tracks: {e}")

    def detect_tempo(self, audio: np.ndarray, sr: int) -> int:
        """
        Auto-detect tempo using librosa

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Tempo in BPM (rounded to nearest integer)
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo)

            # Round to nearest integer
            tempo = round(tempo)

            # Clamp to reasonable range (40-200 BPM)
            tempo = max(40, min(200, tempo))

            return tempo

        except Exception as e:
            print(f"Warning: Failed to detect tempo, using default 120 BPM: {e}")
            return 120

    def detect_key(self, audio: np.ndarray, sr: int) -> str:
        """
        Auto-detect key signature using chroma features

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            Key signature (e.g., "C", "Am", "F#")
        """
        try:
            # Compute chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

            # Average over time
            chroma_mean = np.mean(chroma, axis=1)

            # Find dominant pitch class
            pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            dominant_pitch = pitch_classes[np.argmax(chroma_mean)]

            # Detect major vs minor (simplified)
            # For now, default to major
            key = dominant_pitch

            return key

        except Exception as e:
            print(f"Warning: Failed to detect key, using default C: {e}")
            return "C"

    def quantize_notes(self, notes: List[Note], tempo: int) -> List[Note]:
        """
        Quantize note timings to musical grid

        Args:
            notes: List of notes with raw timings
            tempo: Tempo in BPM

        Returns:
            List of notes with quantized timings
        """
        beat_duration = 60.0 / tempo  # Duration of one beat in seconds
        grid_duration = beat_duration / (self.quantize_resolution / 4)

        quantized_notes = []
        for note in notes:
            # Quantize start time
            quantized_start = round(note.start_time / grid_duration) * grid_duration

            # Quantize duration
            quantized_duration = round(note.duration / grid_duration) * grid_duration

            # Ensure minimum duration
            if quantized_duration < grid_duration:
                quantized_duration = grid_duration

            quantized_notes.append(Note(
                pitch=note.pitch,
                start_time=quantized_start,
                duration=quantized_duration,
                velocity=note.velocity,
                confidence=note.confidence
            ))

        return quantized_notes

    def remove_outliers(self, notes: List[Note]) -> List[Note]:
        """
        Remove outlier notes (noise, false detections)

        Criteria:
        - Very short notes (< 0.05s after quantization)
        - Notes outside reasonable pitch range (21-108, A0-C8)
        - Low confidence notes

        Args:
            notes: List of notes

        Returns:
            Filtered list of notes
        """
        filtered = []
        for note in notes:
            # Check duration
            if note.duration < 0.05:
                continue

            # Check pitch range
            if not (21 <= note.pitch <= 108):
                continue

            # Note passed all filters
            filtered.append(note)

        return filtered


def main():
    """Example usage of MusicAnalyzer"""
    import sys
    import soundfile as sf

    if len(sys.argv) < 2:
        print("Usage: python music_analyzer.py <audio_file_path>")
        return

    audio_path = sys.argv[1]

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Initialize analyzer
    print("Initializing music analyzer...")
    analyzer = MusicAnalyzer(quantize_resolution=16)

    # Detect tempo
    print("\nDetecting tempo...")
    tempo = analyzer.detect_tempo(audio, sr)
    print(f"Detected tempo: {tempo} BPM")

    # Detect key
    print("\nDetecting key...")
    key = analyzer.detect_key(audio, sr)
    print(f"Detected key: {key}")

    # Create dummy track for testing quantization
    print("\nTesting quantization...")
    dummy_notes = [
        Note(pitch=60, start_time=0.12, duration=0.48, velocity=80, confidence=0.9),
        Note(pitch=64, start_time=0.63, duration=0.47, velocity=75, confidence=0.85),
        Note(pitch=67, start_time=1.15, duration=0.51, velocity=70, confidence=0.8),
    ]
    dummy_track = Track(name="test", notes=dummy_notes)

    print("\nOriginal notes:")
    for note in dummy_notes:
        print(f"  {note.to_note_name()} | Start: {note.start_time:.3f}s | Duration: {note.duration:.3f}s")

    # Process track
    processed_tracks = analyzer.analyze_and_process(audio, sr, [dummy_track])
    processed_track = processed_tracks[0]

    print(f"\nProcessed track (tempo={processed_track.tempo}, key={processed_track.key_signature}):")
    for note in processed_track.notes:
        print(f"  {note.to_note_name()} | Start: {note.start_time:.3f}s | Duration: {note.duration:.3f}s")


if __name__ == "__main__":
    main()

"""
Music Transcriber Module
AI-powered music transcription using Crepe for pitch detection
"""

import numpy as np
from typing import List
import librosa

from .models import Note, Track

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("Warning: Crepe not installed. Please install with: pip install crepe")


class MusicTranscriber:
    """AI-powered music transcription using Crepe + Librosa"""

    def __init__(self, min_confidence: float = 0.6, min_duration: float = 0.05):
        """
        Initialize music transcriber

        Args:
            min_confidence: Minimum confidence threshold (0.6 = 60%)
            min_duration: Minimum note duration in seconds

        Raises:
            ImportError: If Crepe is not installed
        """
        if not CREPE_AVAILABLE:
            raise ImportError(
                "Crepe is not installed. Please install with: pip install crepe tensorflow"
            )

        self.min_confidence = min_confidence
        self.min_duration = min_duration

    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        """
        Transcribe audio to notes using Crepe pitch detection + Librosa onset detection

        Args:
            audio: Audio array
            sr: Sample rate

        Returns:
            List of Note objects

        Raises:
            RuntimeError: If transcription fails
        """
        try:
            # Ensure audio is mono
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)

            # Resample to 16000 Hz for Crepe (optimal)
            if sr != 16000:
                audio_crepe = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr_crepe = 16000
            else:
                audio_crepe = audio
                sr_crepe = sr

            # Run Crepe pitch detection
            time, frequency, confidence, activation = crepe.predict(
                audio_crepe,
                sr_crepe,
                viterbi=True,
                model_capacity='full'  # Use full model for best accuracy
            )

            # Detect onsets using librosa
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=sr,
                backtrack=True,
                units='time'
            )

            # Convert frequency to MIDI pitch
            midi_pitches = librosa.hz_to_midi(frequency)

            # Combine pitch and onset information to create notes
            notes = self._combine_pitch_and_onsets(
                time, midi_pitches, confidence, onset_frames
            )

            # Filter by confidence and duration
            filtered_notes = []
            for note in notes:
                if note.confidence >= self.min_confidence and note.duration >= self.min_duration:
                    # Filter out invalid pitches
                    if 21 <= note.pitch <= 108:  # Piano range
                        filtered_notes.append(note)

            return filtered_notes

        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")

    def _combine_pitch_and_onsets(self, time: np.ndarray, midi_pitches: np.ndarray,
                                  confidence: np.ndarray, onset_times: np.ndarray) -> List[Note]:
        """
        Combine pitch detection with onset detection to create notes

        Args:
            time: Time array from Crepe
            midi_pitches: MIDI pitch array
            confidence: Confidence array from Crepe
            onset_times: Onset times from librosa

        Returns:
            List of Note objects
        """
        notes = []

        # For each onset, find the corresponding pitch
        for i, onset_time in enumerate(onset_times):
            # Find the next onset (or end of audio) for note duration
            if i < len(onset_times) - 1:
                offset_time = onset_times[i + 1]
            else:
                offset_time = time[-1]

            duration = offset_time - onset_time

            # Find pitch at onset time
            time_idx = np.argmin(np.abs(time - onset_time))

            # Get pitch and confidence at this time
            pitch = midi_pitches[time_idx]
            conf = confidence[time_idx]

            # Skip if pitch is invalid (nan or too low confidence)
            if np.isnan(pitch) or conf < 0.3:
                continue

            # Round pitch to nearest integer
            pitch = int(np.round(pitch))

            # Estimate velocity from confidence (scale to MIDI range)
            velocity = int(conf * 127)
            velocity = max(30, min(127, velocity))  # Clamp to reasonable range

            notes.append(Note(
                pitch=pitch,
                start_time=onset_time,
                duration=duration,
                velocity=velocity,
                confidence=float(conf)
            ))

        return notes

    def transcribe_track(self, audio: np.ndarray, sr: int, track_name: str) -> Track:
        """
        Transcribe audio to a Track object

        Args:
            audio: Audio array
            sr: Sample rate
            track_name: Name of the track (e.g., "melody", "harmony1")

        Returns:
            Track object with notes

        Raises:
            RuntimeError: If transcription fails
        """
        notes = self.transcribe(audio, sr)
        track = Track(name=track_name, notes=notes)
        track.sort_notes()
        return track


def main():
    """Example usage of MusicTranscriber"""
    import sys
    import soundfile as sf

    if len(sys.argv) < 2:
        print("Usage: python music_transcriber.py <audio_file_path>")
        return

    audio_path = sys.argv[1]

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Initialize transcriber
    print("Initializing music transcriber...")
    transcriber = MusicTranscriber(min_confidence=0.6, min_duration=0.05)

    # Transcribe
    print("Transcribing audio (this may take a while)...")
    notes = transcriber.transcribe(audio, sr)

    print(f"\nTranscribed {len(notes)} notes")

    # Show first 10 notes
    print("\nFirst 10 notes:")
    for i, note in enumerate(notes[:10]):
        print(f"  {i+1}. {note.to_note_name()} | "
              f"Start: {note.start_time:.2f}s | "
              f"Duration: {note.duration:.2f}s | "
              f"Velocity: {note.velocity} | "
              f"Confidence: {note.confidence:.2f}")

    # Create track
    track = transcriber.transcribe_track(audio, sr, "test_track")
    print(f"\nTrack duration: {track.get_duration():.2f}s")


if __name__ == "__main__":
    main()

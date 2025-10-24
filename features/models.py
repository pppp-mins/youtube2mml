"""
Data Models for YouTube2MML
Musical note and track representations
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Note:
    """Musical note representation"""
    pitch: int          # MIDI note number (0-127), 60=C4
    start_time: float   # Start time in seconds
    duration: float     # Duration in seconds
    velocity: int = 64  # Velocity (0-127)
    confidence: float = 1.0  # Transcription confidence (0.0-1.0)

    def to_midi_note(self) -> int:
        """Convert to MIDI note number"""
        return self.pitch

    def to_frequency(self) -> float:
        """Convert to frequency in Hz"""
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))

    def to_note_name(self) -> str:
        """Convert to note name (e.g., 'C4', 'F#5')"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        note = notes[self.pitch % 12]
        return f"{note}{octave}"


@dataclass
class Track:
    """Music track with analyzed notes"""
    name: str                           # "melody", "harmony1", "harmony2"
    notes: List[Note] = field(default_factory=list)
    tempo: int = 120                    # BPM (beats per minute)
    time_signature: Tuple[int, int] = (4, 4)  # (numerator, denominator)
    key_signature: str = "C"            # Key (C, D, E, F, G, A, B + m for minor)
    instrument: Optional[str] = None

    def add_note(self, note: Note):
        """Add a note to the track"""
        self.notes.append(note)

    def sort_notes(self):
        """Sort notes by start time"""
        self.notes.sort(key=lambda n: n.start_time)

    def get_duration(self) -> float:
        """Get total duration of track in seconds"""
        if not self.notes:
            return 0.0
        return max(n.start_time + n.duration for n in self.notes)

    def filter_by_confidence(self, min_confidence: float):
        """Remove low-confidence notes"""
        self.notes = [n for n in self.notes if n.confidence >= min_confidence]


@dataclass
class MusicPiece:
    """Complete transcribed music"""
    title: str
    tracks: List[Track] = field(default_factory=list)
    source_url: Optional[str] = None

    def add_track(self, track: Track):
        """Add a track to the piece"""
        self.tracks.append(track)

    def get_track(self, name: str) -> Optional[Track]:
        """Get track by name"""
        for track in self.tracks:
            if track.name == name:
                return track
        return None

    def get_melody(self) -> Optional[Track]:
        """Get melody track"""
        return self.get_track("melody")

    def get_harmonies(self) -> List[Track]:
        """Get all harmony tracks"""
        return [t for t in self.tracks if t.name.startswith("harmony")]

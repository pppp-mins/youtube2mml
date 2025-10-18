# System Architecture & Implementation Guide

This document provides detailed implementation guidance for converting YouTube audio to MML code.

## Table of Contents
1. [Stage 2: Audio Preprocessor](#stage-2-audio-preprocessor)
2. [Stage 3: Source Separation](#stage-3-source-separation)
3. [Stage 4: Music Transcription](#stage-4-music-transcription)
4. [Implementation Strategies](#implementation-strategies)
5. [Data Structure Design](#data-structure-design)
6. [Decision Checklist](#decision-checklist)

---

## Stage 2: Audio Preprocessor

### Overview
Prepares audio files for analysis by loading, normalizing, and optionally segmenting the audio.

### Implementation

**Recommended Libraries:**
- `librosa`: Audio loading, resampling, analysis
- `soundfile`: Audio file I/O
- `pydub`: Simple audio manipulation (optional)

**Core Class Structure:**
```python
class AudioPreprocessor:
    def __init__(self, sample_rate=22050, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize

    def load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate"""

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude to [-1, 1]"""

    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Remove silence from beginning and end"""

    def segment_audio(self, audio: np.ndarray, segment_length: float) -> list:
        """Split audio into segments of specified length (seconds)"""
```

### Decision Points

#### 1. Sample Rate Selection
- **22050 Hz** (Recommended)
  - Standard for music analysis
  - Fast processing
  - Sufficient for most music transcription tasks
  - Librosa default

- **44100 Hz**
  - CD quality
  - More accurate pitch detection
  - Slower processing
  - Higher memory usage

#### 2. Audio Length Limits
- **Full track**: Process entire song
  - Pros: Complete transcription
  - Cons: Slow, memory-intensive

- **Time-limited** (e.g., 30s, 60s):
  - Pros: Fast prototyping, manageable memory
  - Cons: Incomplete transcription

- **Segment-based**: Process in chunks
  - Pros: Memory efficient, parallelizable
  - Cons: Requires merging logic

#### 3. Preprocessing Intensity
- **Noise reduction**: Apply spectral gating?
- **Silence trimming**: Remove silent sections?
- **Volume normalization**: Peak vs RMS normalization?
- **Resampling**: Target sample rate conversion?

---

## Stage 3: Source Separation

### Overview
Separates audio into melody and harmony tracks using AI models.

### Implementation Options

#### Option A: Demucs (Meta AI) - **Recommended for Quality**

```python
from demucs import pretrained
from demucs.apply import apply_model
import torch

class SourceSeparator:
    def __init__(self, model_name='htdemucs'):
        self.model = pretrained.get_model(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def separate(self, audio_path: str) -> dict:
        """
        Separate audio into stems
        Returns: {'vocals': audio, 'drums': audio, 'bass': audio, 'other': audio}
        """
```

**Models:**
- `htdemucs`: Highest quality, slowest, GPU recommended
- `htdemucs_ft`: Fine-tuned version, good balance
- `mdx_extra`: Fast, good quality

**Pros:**
- State-of-the-art quality
- Actively maintained
- Multiple model options

**Cons:**
- Heavy (GPU recommended)
- Slower processing
- Large model download

#### Option B: Spleeter (Deezer) - **Recommended for Speed**

```python
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

class SourceSeparator:
    def __init__(self, stems=4):
        self.separator = Separator(f'spleeter:{stems}stems')

    def separate(self, audio_path: str) -> dict:
        """
        Separate audio into stems
        stems=2: vocals, accompaniment
        stems=4: vocals, drums, bass, other
        stems=5: vocals, drums, bass, piano, other
        """
```

**Pros:**
- Fast processing
- Works well on CPU
- Easy to use

**Cons:**
- Lower quality than Demucs
- Less actively maintained
- Limited to predefined stem counts

#### Option C: Open-Unmix

```python
import openunmix

class SourceSeparator:
    def __init__(self):
        self.separator = openunmix.umx.OpenUnmix()

    def separate(self, audio_path: str) -> dict:
        """Lightweight separation"""
```

**Pros:**
- Lightweight
- Fast

**Cons:**
- Lower quality
- Less flexible

### Decision Points

#### 1. Model Selection

| Model | Quality | Speed | GPU Required | Memory | Use Case |
|-------|---------|-------|--------------|--------|----------|
| Demucs htdemucs | Excellent | Slow | Recommended | High | Production, high quality |
| Demucs mdx_extra | Good | Medium | Optional | Medium | Balanced |
| Spleeter 4-stem | Good | Fast | No | Low | Prototyping, CPU-only |
| Spleeter 2-stem | Fair | Very Fast | No | Low | Quick testing |

#### 2. Track Separation Strategy

**Option A: Stem-based Separation**
- 2-stem: vocals + accompaniment
- 4-stem: vocals, drums, bass, other
- 5-stem: vocals, drums, bass, piano, other

**Option B: Custom Track Assignment**
```
Melody Track: vocals OR lead instrument from 'other'
Harmony 1: bass
Harmony 2: drums (rhythm converted to pitched percussion)
Harmony 3: other (chords/pads)
```

#### 3. Melody vs Harmony Definition

**Strategy 1: Stem-to-Track Mapping**
```
vocals → Melody
bass → Harmony 1
other → Harmony 2
drums → (optional) Harmony 3 or ignore
```

**Strategy 2: Post-separation Analysis**
```
All stems → Pitch detection → Separate by pitch range
  High register → Melody
  Mid register → Harmony 1, 2
  Low register → Harmony 3 (bass)
```

#### 4. Processing Resources
- **GPU available**: Use Demucs htdemucs
- **CPU only**: Use Spleeter or Demucs with CPU optimization
- **Memory limited**: Process audio in segments
- **Speed critical**: Use Spleeter 2-stem or 4-stem

---

## Stage 4: Music Transcription

### Overview
Converts audio signals to musical note data (pitch, duration, timing).

### Implementation Options

#### Option A: Basic-Pitch (Spotify) - **Recommended**

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

class MusicTranscriber:
    def __init__(self):
        pass

    def transcribe(self, audio_path: str) -> list:
        """
        Transcribe audio to note events
        Returns: [(start_time, end_time, pitch, velocity, confidence)]
        """
        model_output, midi_data, note_events = predict(audio_path)
        return note_events
```

**Pros:**
- Polyphonic (handles chords)
- Direct MIDI output
- Good accuracy
- Easy to use

**Cons:**
- Fixed model (no fine-tuning)
- Moderate speed

#### Option B: Crepe (Pitch Detection) + Onset Detection

```python
import crepe
import librosa

class MusicTranscriber:
    def transcribe(self, audio: np.ndarray, sr: int) -> list:
        # Pitch detection
        time, frequency, confidence, activation = crepe.predict(
            audio, sr, viterbi=True
        )

        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Combine pitch + onset → notes
        notes = self._combine_pitch_and_onsets(time, frequency, onset_times)
        return notes
```

**Pros:**
- Very accurate pitch detection
- Customizable
- Fine control

**Cons:**
- Monophonic only (one note at a time)
- Requires manual onset/offset detection
- More complex implementation

#### Option C: Librosa (Traditional Signal Processing)

```python
import librosa

class MusicTranscriber:
    def transcribe(self, audio: np.ndarray, sr: int) -> list:
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)

        # Extract notes
        notes = self._extract_notes(pitches, magnitudes, onset_frames, sr)
        return notes
```

**Pros:**
- Lightweight
- No additional dependencies
- Highly customizable

**Cons:**
- Lower accuracy
- Requires more manual tuning
- Monophonic

#### Option D: MT3 (Music Transformer) - Advanced

```python
# Google's transformer-based music transcription
# Most accurate but very heavy
```

**Pros:**
- State-of-the-art accuracy
- Polyphonic
- Handles complex music

**Cons:**
- Very slow
- Requires GPU
- Complex setup
- Large model

### Decision Points

#### 1. Transcription Method Selection

| Method | Polyphonic | Accuracy | Speed | Complexity | Use Case |
|--------|------------|----------|-------|------------|----------|
| Basic-Pitch | Yes | Good | Medium | Low | Recommended for most cases |
| Crepe + Onset | No | Excellent | Fast | High | Melody only, high precision |
| Librosa | No | Fair | Very Fast | Medium | Prototyping, lightweight |
| MT3 | Yes | Excellent | Slow | High | Production, complex music |

#### 2. Polyphonic Handling

**For Melody Track (usually monophonic):**
- Use pitch detection with onset detection
- If multiple pitches detected, pick highest or loudest

**For Harmony Tracks (can be polyphonic):**
- Use Basic-Pitch or MT3 for chord detection
- Or treat each harmony as monophonic (one note at a time)

#### 3. Note Quantization

**Purpose:** Align detected note timings to musical grid

**Quantization Resolution:**
- **1/4 notes**: Simple melodies, slow tempo
- **1/8 notes**: Standard, works for most music
- **1/16 notes**: Complex rhythms, fast passages
- **1/32 notes**: Very detailed, rare

**Implementation:**
```python
def quantize_time(time: float, bpm: int, resolution: int = 16) -> float:
    """
    Quantize time to nearest beat division
    resolution: 4=quarter, 8=eighth, 16=sixteenth
    """
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration / (resolution / 4)
    return round(time / grid_duration) * grid_duration
```

**Options:**
- **No quantization**: Keep exact timings (human feel)
- **Soft quantization**: Nudge towards grid but keep some variation
- **Hard quantization**: Snap to exact grid

#### 4. Noise Filtering

**Confidence Threshold:**
```python
min_confidence = 0.5  # Discard notes with confidence < 0.5
notes = [n for n in notes if n.confidence >= min_confidence]
```

**Minimum Note Duration:**
```python
min_duration = 0.1  # seconds
notes = [n for n in notes if n.duration >= min_duration]
```

**Pitch Range Limiting:**
```python
min_pitch = 21  # A0
max_pitch = 108  # C8
notes = [n for n in notes if min_pitch <= n.pitch <= max_pitch]
```

#### 5. Tempo & Time Signature Detection

**Option A: Automatic Detection**
```python
import librosa

def detect_tempo(audio: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo

def detect_time_signature(audio: np.ndarray, sr: int) -> tuple:
    # More complex, often requires ML models
    # Or use madmom library
    return (4, 4)  # default
```

**Option B: Fixed Values**
```python
DEFAULT_TEMPO = 120  # BPM
DEFAULT_TIME_SIGNATURE = (4, 4)
```

**Option C: User Input**
- Prompt user to specify tempo and time signature
- More accurate but requires user knowledge

---

## Implementation Strategies

### Strategy 1: Fast Prototype (CPU, Quick Results)

**Pipeline:**
```
Audio Preprocessor (librosa, 22050 Hz)
  ↓
Spleeter (2-stem or 4-stem)
  ↓
Basic-Pitch (per stem)
  ↓
Simple quantization (1/8 notes)
  ↓
MML Generator
```

**Dependencies:**
```txt
librosa==0.10.1
soundfile==0.12.1
spleeter==2.4.0
basic-pitch==0.2.5
numpy==1.24.3
```

**Pros:**
- Fast setup
- Works on CPU
- Good for testing

**Cons:**
- Lower quality separation
- May miss subtle details

### Strategy 2: High Quality (GPU, Best Results)

**Pipeline:**
```
Audio Preprocessor (librosa, 44100 Hz)
  ↓
Demucs htdemucs (4-stem or 6-stem)
  ↓
Basic-Pitch or MT3 (per stem)
  ↓
Advanced quantization with swing detection
  ↓
MML Generator
```

**Dependencies:**
```txt
librosa==0.10.1
soundfile==0.12.1
demucs==4.0.1
basic-pitch==0.2.5
torch==2.0.1
torchaudio==2.0.2
numpy==1.24.3
```

**Pros:**
- Best quality output
- More accurate transcription
- Better separation

**Cons:**
- Requires GPU
- Slower processing
- More memory

### Strategy 3: Balanced (Recommended for Production)

**Pipeline:**
```
Audio Preprocessor (librosa, 22050 Hz)
  ↓
Demucs mdx_extra (4-stem)
  ↓
Basic-Pitch (per stem)
  ↓
Adaptive quantization
  ↓
MML Generator
```

**Dependencies:**
```txt
librosa==0.10.1
soundfile==0.12.1
demucs==4.0.1
basic-pitch==0.2.5
torch==2.0.1
numpy==1.24.3
```

**Pros:**
- Good quality
- Reasonable speed
- Works with or without GPU

---

## Data Structure Design

### Note Representation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Note:
    """Represents a single musical note"""
    pitch: int  # MIDI note number (0-127), 60=C4
    start_time: float  # seconds from start
    duration: float  # seconds
    velocity: int = 64  # 0-127, loudness
    confidence: float = 1.0  # 0.0-1.0, transcription confidence

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
```

### Track Representation

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Track:
    """Represents a musical track (melody or harmony)"""
    name: str  # "melody", "harmony1", "harmony2", etc.
    notes: List[Note] = field(default_factory=list)
    tempo: int = 120  # BPM
    time_signature: tuple = (4, 4)  # (numerator, denominator)
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
```

### Complete Music Piece

```python
@dataclass
class MusicPiece:
    """Represents complete transcribed music"""
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
```

---

## Decision Checklist

Before implementing, answer these questions:

### System Resources
- [ ] GPU available? (Y/N)
- [ ] Target processing time? (seconds per song)
- [ ] Memory constraints? (GB available)

### Quality Requirements
- [ ] Speed vs Quality priority? (Fast/Balanced/Quality)
- [ ] Acceptable transcription accuracy? (70%/80%/90%+)
- [ ] Number of harmony tracks needed? (0-5)

### Audio Processing
- [ ] Sample rate? (22050/44100 Hz)
- [ ] Process full track or time-limited? (Full/30s/60s/custom)
- [ ] Remove silence? (Y/N)
- [ ] Normalize audio? (Y/N)

### Source Separation
- [ ] Separation model? (Demucs/Spleeter/Other)
- [ ] Number of stems? (2/4/5/6)
- [ ] Melody definition? (Vocals/Lead instrument/Both)
- [ ] Harmony mapping strategy? (Stem-based/Analysis-based)

### Music Transcription
- [ ] Transcription method? (Basic-Pitch/Crepe/MT3/Librosa)
- [ ] Handle polyphonic harmonies? (Y/N)
- [ ] Quantization resolution? (None/1/4/1/8/1/16)
- [ ] Minimum note confidence? (0.3-0.8)
- [ ] Minimum note duration? (0.05-0.2 seconds)

### Tempo & Time Signature
- [ ] Automatic tempo detection? (Y/N/User input)
- [ ] Default tempo if detection fails? (120 BPM)
- [ ] Automatic time signature detection? (Y/N/User input)
- [ ] Default time signature? ((4,4))

---

## Recommended Implementation Order

### Phase 1: Basic Implementation (1-2 days)

1. **Audio Preprocessor**
   - Implement `AudioPreprocessor` class
   - Test with sample audio files
   - Verify audio loading and normalization

2. **Source Separation**
   - Install Spleeter (easiest to start)
   - Implement `SourceSeparator` class
   - Test separation on sample files
   - Verify output quality

3. **Music Transcription**
   - Install Basic-Pitch
   - Implement `MusicTranscriber` class
   - Test transcription on separated stems
   - Verify note extraction

4. **Data Structures**
   - Implement `Note`, `Track`, `MusicPiece` classes
   - Add serialization methods (to JSON/dict)
   - Write unit tests

### Phase 2: Integration & Testing (1-2 days)

1. **Pipeline Integration**
   - Create end-to-end pipeline class
   - Connect all components
   - Handle errors gracefully

2. **Test Suite**
   - Test with various music genres
   - Test with different audio qualities
   - Test edge cases (silence, noise, etc.)

3. **Validation**
   - Visual inspection of results
   - Compare with ground truth (if available)
   - Identify failure modes

### Phase 3: Optimization & Enhancement (2-3 days)

1. **Performance Optimization**
   - Profile bottlenecks
   - Implement caching
   - Parallelize where possible

2. **Quality Improvements**
   - Upgrade to Demucs if needed
   - Fine-tune transcription parameters
   - Implement advanced quantization

3. **Additional Features**
   - Add progress callbacks
   - Implement batch processing
   - Add configuration file support

---

## Example Configuration File

```yaml
# config.yaml
audio:
  sample_rate: 22050
  normalize: true
  trim_silence: true
  max_duration: null  # null = no limit, or seconds

separation:
  model: "spleeter"  # or "demucs"
  stems: 4
  model_name: "htdemucs"  # for demucs
  device: "auto"  # "cuda", "cpu", or "auto"

transcription:
  method: "basic-pitch"  # or "crepe", "librosa"
  min_confidence: 0.5
  min_duration: 0.1
  pitch_range: [21, 108]  # A0 to C8

quantization:
  enabled: true
  resolution: 16  # 16th notes
  strength: 0.8  # 0.0=none, 1.0=hard snap

tracks:
  melody_stem: "vocals"  # which stem to use for melody
  harmonies:
    - name: "harmony1"
      stem: "bass"
    - name: "harmony2"
      stem: "other"

tempo:
  auto_detect: true
  default: 120

time_signature:
  auto_detect: false
  default: [4, 4]
```

---

## Next Steps

1. Answer the decision checklist questions
2. Choose implementation strategy (Fast/Balanced/Quality)
3. Set up development environment
4. Implement Phase 1 components
5. Test and iterate

## Questions?

If you need help deciding on any of these points, consider:
- What is your primary use case? (prototyping/production/research)
- What is your target audience? (developers/musicians/end-users)
- What are your hardware constraints?
- What is your timeline?

Based on your answers, we can recommend the optimal implementation approach.

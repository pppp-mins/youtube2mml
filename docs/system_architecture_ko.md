# 시스템 아키텍처 및 구현 가이드

이 문서는 YouTube 음원을 MML 코드로 변환하는 상세한 구현 가이드를 제공합니다.

## 목차
1. [2단계: 오디오 전처리기](#2단계-오디오-전처리기)
2. [3단계: 음원 분리](#3단계-음원-분리)
3. [4단계: 음악 전사](#4단계-음악-전사)
4. [구현 전략](#구현-전략)
5. [데이터 구조 설계](#데이터-구조-설계)
6. [의사결정 체크리스트](#의사결정-체크리스트)

---

## 2단계: 오디오 전처리기

### 개요
오디오 파일을 로드하고, 정규화하며, 필요시 세그먼트로 분할하여 분석을 위한 준비를 합니다.

### 구현

**추천 라이브러리:**
- `librosa`: 오디오 로딩, 리샘플링, 분석
- `soundfile`: 오디오 파일 I/O
- `pydub`: 간단한 오디오 조작 (선택사항)

**핵심 클래스 구조:**
```python
class AudioPreprocessor:
    def __init__(self, sample_rate=22050, normalize=True):
        self.sample_rate = sample_rate
        self.normalize = normalize

    def load_audio(self, file_path: str) -> tuple[np.ndarray, int]:
        """오디오 파일을 로드하고 오디오 데이터와 샘플레이트 반환"""

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 진폭을 [-1, 1]로 정규화"""

    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """시작과 끝의 무음 제거"""

    def segment_audio(self, audio: np.ndarray, segment_length: float) -> list:
        """오디오를 지정된 길이(초)의 세그먼트로 분할"""
```

### 의사결정 사항

#### 1. 샘플레이트 선택
- **22050 Hz** (추천)
  - 음악 분석의 표준
  - 빠른 처리
  - 대부분의 음악 전사 작업에 충분
  - Librosa 기본값

- **44100 Hz**
  - CD 음질
  - 더 정확한 피치 감지
  - 느린 처리
  - 높은 메모리 사용량

#### 2. 오디오 길이 제한
- **전체 트랙**: 전체 곡 처리
  - 장점: 완전한 전사
  - 단점: 느림, 메모리 집약적

- **시간 제한** (예: 30초, 60초):
  - 장점: 빠른 프로토타이핑, 관리 가능한 메모리
  - 단점: 불완전한 전사

- **세그먼트 기반**: 청크 단위로 처리
  - 장점: 메모리 효율적, 병렬화 가능
  - 단점: 병합 로직 필요

#### 3. 전처리 강도
- **노이즈 감소**: 스펙트럴 게이팅 적용?
- **무음 제거**: 무음 구간 제거?
- **볼륨 정규화**: 피크 vs RMS 정규화?
- **리샘플링**: 목표 샘플레이트 변환?

---

## 3단계: 음원 분리

### 개요
AI 모델을 사용하여 오디오를 멜로디와 화음 트랙으로 분리합니다.

### 구현 옵션

#### 옵션 A: Demucs (Meta AI) - **품질 우선 추천**

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
        오디오를 스템으로 분리
        반환값: {'vocals': audio, 'drums': audio, 'bass': audio, 'other': audio}
        """
```

**모델:**
- `htdemucs`: 최고 품질, 가장 느림, GPU 권장
- `htdemucs_ft`: 파인튜닝 버전, 좋은 균형
- `mdx_extra`: 빠름, 좋은 품질

**장점:**
- 최첨단 품질
- 활발한 유지보수
- 다양한 모델 옵션

**단점:**
- 무거움 (GPU 권장)
- 느린 처리
- 큰 모델 다운로드

#### 옵션 B: Spleeter (Deezer) - **속도 우선 추천**

```python
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

class SourceSeparator:
    def __init__(self, stems=4):
        self.separator = Separator(f'spleeter:{stems}stems')

    def separate(self, audio_path: str) -> dict:
        """
        오디오를 스템으로 분리
        stems=2: vocals, accompaniment
        stems=4: vocals, drums, bass, other
        stems=5: vocals, drums, bass, piano, other
        """
```

**장점:**
- 빠른 처리
- CPU에서도 잘 작동
- 사용하기 쉬움

**단점:**
- Demucs보다 낮은 품질
- 덜 활발한 유지보수
- 사전 정의된 스템 수로 제한

#### 옵션 C: Open-Unmix

```python
import openunmix

class SourceSeparator:
    def __init__(self):
        self.separator = openunmix.umx.OpenUnmix()

    def separate(self, audio_path: str) -> dict:
        """경량 분리"""
```

**장점:**
- 경량
- 빠름

**단점:**
- 낮은 품질
- 덜 유연함

### 의사결정 사항

#### 1. 모델 선택

| 모델 | 품질 | 속도 | GPU 필요 | 메모리 | 사용 사례 |
|------|------|------|----------|--------|-----------|
| Demucs htdemucs | 우수 | 느림 | 권장 | 높음 | 프로덕션, 고품질 |
| Demucs mdx_extra | 좋음 | 보통 | 선택 | 보통 | 균형잡힌 |
| Spleeter 4-stem | 좋음 | 빠름 | 아니오 | 낮음 | 프로토타이핑, CPU 전용 |
| Spleeter 2-stem | 보통 | 매우 빠름 | 아니오 | 낮음 | 빠른 테스트 |

#### 2. 트랙 분리 전략

**옵션 A: 스템 기반 분리**
- 2-stem: vocals + accompaniment
- 4-stem: vocals, drums, bass, other
- 5-stem: vocals, drums, bass, piano, other

**옵션 B: 커스텀 트랙 할당**
```
멜로디 트랙: vocals 또는 'other'에서 리드 악기
화음 1: bass
화음 2: drums (리듬을 피치 있는 타악기로 변환)
화음 3: other (코드/패드)
```

#### 3. 멜로디 vs 화음 정의

**전략 1: 스템-트랙 매핑**
```
vocals → 멜로디
bass → 화음 1
other → 화음 2
drums → (선택적) 화음 3 또는 무시
```

**전략 2: 분리 후 분석**
```
모든 스템 → 피치 감지 → 피치 범위로 분리
  고음역 → 멜로디
  중음역 → 화음 1, 2
  저음역 → 화음 3 (베이스)
```

#### 4. 처리 리소스
- **GPU 사용 가능**: Demucs htdemucs 사용
- **CPU 전용**: Spleeter 또는 CPU 최적화된 Demucs 사용
- **메모리 제한**: 오디오를 세그먼트로 처리
- **속도 중요**: Spleeter 2-stem 또는 4-stem 사용

---

## 4단계: 음악 전사

### 개요
오디오 신호를 음표 데이터(피치, 지속시간, 타이밍)로 변환합니다.

### 구현 옵션

#### 옵션 A: Basic-Pitch (Spotify) - **추천**

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

class MusicTranscriber:
    def __init__(self):
        pass

    def transcribe(self, audio_path: str) -> list:
        """
        오디오를 음표 이벤트로 전사
        반환값: [(start_time, end_time, pitch, velocity, confidence)]
        """
        model_output, midi_data, note_events = predict(audio_path)
        return note_events
```

**장점:**
- 폴리포닉 (화음 처리)
- 직접 MIDI 출력
- 좋은 정확도
- 사용하기 쉬움

**단점:**
- 고정 모델 (파인튜닝 불가)
- 보통 속도

#### 옵션 B: Crepe (피치 감지) + 온셋 감지

```python
import crepe
import librosa

class MusicTranscriber:
    def transcribe(self, audio: np.ndarray, sr: int) -> list:
        # 피치 감지
        time, frequency, confidence, activation = crepe.predict(
            audio, sr, viterbi=True
        )

        # 온셋 감지
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # 피치 + 온셋 결합 → 음표
        notes = self._combine_pitch_and_onsets(time, frequency, onset_times)
        return notes
```

**장점:**
- 매우 정확한 피치 감지
- 커스터마이징 가능
- 세밀한 제어

**단점:**
- 모노포닉만 가능 (한 번에 하나의 음표)
- 수동 온셋/오프셋 감지 필요
- 더 복잡한 구현

#### 옵션 C: Librosa (전통적 신호 처리)

```python
import librosa

class MusicTranscriber:
    def transcribe(self, audio: np.ndarray, sr: int) -> list:
        # 피치 추적
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)

        # 온셋 감지
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)

        # 음표 추출
        notes = self._extract_notes(pitches, magnitudes, onset_frames, sr)
        return notes
```

**장점:**
- 경량
- 추가 의존성 없음
- 높은 커스터마이징

**단점:**
- 낮은 정확도
- 더 많은 수동 튜닝 필요
- 모노포닉

#### 옵션 D: MT3 (Music Transformer) - 고급

```python
# Google의 트랜스포머 기반 음악 전사
# 가장 정확하지만 매우 무거움
```

**장점:**
- 최첨단 정확도
- 폴리포닉
- 복잡한 음악 처리

**단점:**
- 매우 느림
- GPU 필요
- 복잡한 설정
- 큰 모델

### 의사결정 사항

#### 1. 전사 방법 선택

| 방법 | 폴리포닉 | 정확도 | 속도 | 복잡도 | 사용 사례 |
|------|----------|--------|------|--------|-----------|
| Basic-Pitch | 예 | 좋음 | 보통 | 낮음 | 대부분의 경우 추천 |
| Crepe + Onset | 아니오 | 우수 | 빠름 | 높음 | 멜로디만, 높은 정밀도 |
| Librosa | 아니오 | 보통 | 매우 빠름 | 보통 | 프로토타이핑, 경량 |
| MT3 | 예 | 우수 | 느림 | 높음 | 프로덕션, 복잡한 음악 |

#### 2. 폴리포닉 처리

**멜로디 트랙 (보통 모노포닉):**
- 온셋 감지와 함께 피치 감지 사용
- 여러 피치가 감지되면 가장 높거나 가장 큰 것 선택

**화음 트랙 (폴리포닉 가능):**
- 코드 감지를 위해 Basic-Pitch 또는 MT3 사용
- 또는 각 화음을 모노포닉으로 처리 (한 번에 하나의 음표)

#### 3. 음표 양자화

**목적:** 감지된 음표 타이밍을 음악적 그리드에 정렬

**양자화 해상도:**
- **1/4 음표**: 간단한 멜로디, 느린 템포
- **1/8 음표**: 표준, 대부분의 음악에 작동
- **1/16 음표**: 복잡한 리듬, 빠른 구절
- **1/32 음표**: 매우 세밀함, 드묾

**구현:**
```python
def quantize_time(time: float, bpm: int, resolution: int = 16) -> float:
    """
    시간을 가장 가까운 박자 분할로 양자화
    resolution: 4=4분음표, 8=8분음표, 16=16분음표
    """
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration / (resolution / 4)
    return round(time / grid_duration) * grid_duration
```

**옵션:**
- **양자화 없음**: 정확한 타이밍 유지 (인간적 느낌)
- **소프트 양자화**: 그리드로 넛지하되 일부 변화 유지
- **하드 양자화**: 정확한 그리드에 스냅

#### 4. 노이즈 필터링

**신뢰도 임계값:**
```python
min_confidence = 0.5  # 신뢰도 < 0.5인 음표 버림
notes = [n for n in notes if n.confidence >= min_confidence]
```

**최소 음표 지속시간:**
```python
min_duration = 0.1  # 초
notes = [n for n in notes if n.duration >= min_duration]
```

**피치 범위 제한:**
```python
min_pitch = 21  # A0
max_pitch = 108  # C8
notes = [n for n in notes if min_pitch <= n.pitch <= max_pitch]
```

#### 5. 템포 및 박자 감지

**옵션 A: 자동 감지**
```python
import librosa

def detect_tempo(audio: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return tempo

def detect_time_signature(audio: np.ndarray, sr: int) -> tuple:
    # 더 복잡함, 종종 ML 모델 필요
    # 또는 madmom 라이브러리 사용
    return (4, 4)  # 기본값
```

**옵션 B: 고정값**
```python
DEFAULT_TEMPO = 120  # BPM
DEFAULT_TIME_SIGNATURE = (4, 4)
```

**옵션 C: 사용자 입력**
- 사용자에게 템포와 박자 지정 요청
- 더 정확하지만 사용자 지식 필요

---

## 구현 전략

### 전략 1: 빠른 프로토타입 (CPU, 빠른 결과)

**파이프라인:**
```
Audio Preprocessor (librosa, 22050 Hz)
  ↓
Spleeter (2-stem 또는 4-stem)
  ↓
Basic-Pitch (스템별)
  ↓
간단한 양자화 (1/8 음표)
  ↓
MML Generator
```

**의존성:**
```txt
librosa==0.10.1
soundfile==0.12.1
spleeter==2.4.0
basic-pitch==0.2.5
numpy==1.24.3
```

**장점:**
- 빠른 설정
- CPU에서 작동
- 테스트에 좋음

**단점:**
- 낮은 품질의 분리
- 미묘한 디테일을 놓칠 수 있음

### 전략 2: 고품질 (GPU, 최고 결과)

**파이프라인:**
```
Audio Preprocessor (librosa, 44100 Hz)
  ↓
Demucs htdemucs (4-stem 또는 6-stem)
  ↓
Basic-Pitch 또는 MT3 (스템별)
  ↓
스윙 감지를 포함한 고급 양자화
  ↓
MML Generator
```

**의존성:**
```txt
librosa==0.10.1
soundfile==0.12.1
demucs==4.0.1
basic-pitch==0.2.5
torch==2.0.1
torchaudio==2.0.2
numpy==1.24.3
```

**장점:**
- 최고 품질 출력
- 더 정확한 전사
- 더 나은 분리

**단점:**
- GPU 필요
- 느린 처리
- 더 많은 메모리

### 전략 3: 균형잡힌 (프로덕션 추천)

**파이프라인:**
```
Audio Preprocessor (librosa, 22050 Hz)
  ↓
Demucs mdx_extra (4-stem)
  ↓
Basic-Pitch (스템별)
  ↓
적응형 양자화
  ↓
MML Generator
```

**의존성:**
```txt
librosa==0.10.1
soundfile==0.12.1
demucs==4.0.1
basic-pitch==0.2.5
torch==2.0.1
numpy==1.24.3
```

**장점:**
- 좋은 품질
- 합리적인 속도
- GPU 유무에 관계없이 작동

---

## 데이터 구조 설계

### 음표 표현

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Note:
    """단일 음표를 나타냄"""
    pitch: int  # MIDI 음표 번호 (0-127), 60=C4
    start_time: float  # 시작부터의 초
    duration: float  # 초
    velocity: int = 64  # 0-127, 음량
    confidence: float = 1.0  # 0.0-1.0, 전사 신뢰도

    def to_midi_note(self) -> int:
        """MIDI 음표 번호로 변환"""
        return self.pitch

    def to_frequency(self) -> float:
        """Hz 단위 주파수로 변환"""
        return 440.0 * (2.0 ** ((self.pitch - 69) / 12.0))

    def to_note_name(self) -> str:
        """음표 이름으로 변환 (예: 'C4', 'F#5')"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        note = notes[self.pitch % 12]
        return f"{note}{octave}"
```

### 트랙 표현

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Track:
    """음악 트랙(멜로디 또는 화음)을 나타냄"""
    name: str  # "melody", "harmony1", "harmony2", 등
    notes: List[Note] = field(default_factory=list)
    tempo: int = 120  # BPM
    time_signature: tuple = (4, 4)  # (분자, 분모)
    instrument: Optional[str] = None

    def add_note(self, note: Note):
        """트랙에 음표 추가"""
        self.notes.append(note)

    def sort_notes(self):
        """시작 시간으로 음표 정렬"""
        self.notes.sort(key=lambda n: n.start_time)

    def get_duration(self) -> float:
        """트랙의 총 지속시간(초) 가져오기"""
        if not self.notes:
            return 0.0
        return max(n.start_time + n.duration for n in self.notes)

    def filter_by_confidence(self, min_confidence: float):
        """낮은 신뢰도의 음표 제거"""
        self.notes = [n for n in self.notes if n.confidence >= min_confidence]
```

### 완전한 음악 작품

```python
@dataclass
class MusicPiece:
    """완전히 전사된 음악을 나타냄"""
    title: str
    tracks: List[Track] = field(default_factory=list)
    source_url: Optional[str] = None

    def add_track(self, track: Track):
        """작품에 트랙 추가"""
        self.tracks.append(track)

    def get_track(self, name: str) -> Optional[Track]:
        """이름으로 트랙 가져오기"""
        for track in self.tracks:
            if track.name == name:
                return track
        return None

    def get_melody(self) -> Optional[Track]:
        """멜로디 트랙 가져오기"""
        return self.get_track("melody")

    def get_harmonies(self) -> List[Track]:
        """모든 화음 트랙 가져오기"""
        return [t for t in self.tracks if t.name.startswith("harmony")]
```

---

## 의사결정 체크리스트

구현하기 전에 다음 질문에 답하세요:

### 시스템 리소스
- [ ] GPU 사용 가능? (예/아니오)
- [ ] 목표 처리 시간? (곡당 초)
- [ ] 메모리 제약? (사용 가능한 GB)

### 품질 요구사항
- [ ] 속도 vs 품질 우선순위? (빠름/균형/품질)
- [ ] 허용 가능한 전사 정확도? (70%/80%/90%+)
- [ ] 필요한 화음 트랙 수? (0-5)

### 오디오 처리
- [ ] 샘플레이트? (22050/44100 Hz)
- [ ] 전체 트랙 처리 또는 시간 제한? (전체/30초/60초/커스텀)
- [ ] 무음 제거? (예/아니오)
- [ ] 오디오 정규화? (예/아니오)

### 음원 분리
- [ ] 분리 모델? (Demucs/Spleeter/기타)
- [ ] 스템 수? (2/4/5/6)
- [ ] 멜로디 정의? (보컬/리드 악기/둘 다)
- [ ] 화음 매핑 전략? (스템 기반/분석 기반)

### 음악 전사
- [ ] 전사 방법? (Basic-Pitch/Crepe/MT3/Librosa)
- [ ] 폴리포닉 화음 처리? (예/아니오)
- [ ] 양자화 해상도? (없음/1/4/1/8/1/16)
- [ ] 최소 음표 신뢰도? (0.3-0.8)
- [ ] 최소 음표 지속시간? (0.05-0.2초)

### 템포 및 박자
- [ ] 자동 템포 감지? (예/아니오/사용자 입력)
- [ ] 감지 실패 시 기본 템포? (120 BPM)
- [ ] 자동 박자 감지? (예/아니오/사용자 입력)
- [ ] 기본 박자? ((4,4))

---

## 추천 구현 순서

### Phase 1: 기본 구현 (1-2일)

1. **오디오 전처리기**
   - `AudioPreprocessor` 클래스 구현
   - 샘플 오디오 파일로 테스트
   - 오디오 로딩 및 정규화 검증

2. **음원 분리**
   - Spleeter 설치 (시작하기 가장 쉬움)
   - `SourceSeparator` 클래스 구현
   - 샘플 파일로 분리 테스트
   - 출력 품질 검증

3. **음악 전사**
   - Basic-Pitch 설치
   - `MusicTranscriber` 클래스 구현
   - 분리된 스템으로 전사 테스트
   - 음표 추출 검증

4. **데이터 구조**
   - `Note`, `Track`, `MusicPiece` 클래스 구현
   - 직렬화 메서드 추가 (JSON/dict로)
   - 단위 테스트 작성

### Phase 2: 통합 및 테스트 (1-2일)

1. **파이프라인 통합**
   - 엔드투엔드 파이프라인 클래스 생성
   - 모든 컴포넌트 연결
   - 에러를 우아하게 처리

2. **테스트 스위트**
   - 다양한 음악 장르로 테스트
   - 다양한 오디오 품질로 테스트
   - 엣지 케이스 테스트 (무음, 노이즈 등)

3. **검증**
   - 결과의 시각적 검사
   - 정답과 비교 (가능한 경우)
   - 실패 모드 식별

### Phase 3: 최적화 및 개선 (2-3일)

1. **성능 최적화**
   - 병목 지점 프로파일링
   - 캐싱 구현
   - 가능한 곳 병렬화

2. **품질 개선**
   - 필요시 Demucs로 업그레이드
   - 전사 파라미터 미세 조정
   - 고급 양자화 구현

3. **추가 기능**
   - 진행 콜백 추가
   - 배치 처리 구현
   - 설정 파일 지원 추가

---

## 예시 설정 파일

```yaml
# config.yaml
audio:
  sample_rate: 22050
  normalize: true
  trim_silence: true
  max_duration: null  # null = 제한 없음, 또는 초

separation:
  model: "spleeter"  # 또는 "demucs"
  stems: 4
  model_name: "htdemucs"  # demucs용
  device: "auto"  # "cuda", "cpu", 또는 "auto"

transcription:
  method: "basic-pitch"  # 또는 "crepe", "librosa"
  min_confidence: 0.5
  min_duration: 0.1
  pitch_range: [21, 108]  # A0부터 C8까지

quantization:
  enabled: true
  resolution: 16  # 16분음표
  strength: 0.8  # 0.0=없음, 1.0=하드 스냅

tracks:
  melody_stem: "vocals"  # 멜로디에 사용할 스템
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

## 다음 단계

1. 의사결정 체크리스트 질문에 답변
2. 구현 전략 선택 (빠름/균형/품질)
3. 개발 환경 설정
4. Phase 1 컴포넌트 구현
5. 테스트 및 반복

## 질문이 있나요?

이 사항들 중 어느 것이든 결정하는 데 도움이 필요하다면 다음을 고려하세요:
- 주요 사용 사례는 무엇인가요? (프로토타이핑/프로덕션/연구)
- 대상 사용자는 누구인가요? (개발자/음악가/최종 사용자)
- 하드웨어 제약 사항은 무엇인가요?
- 타임라인은 어떻게 되나요?

답변에 따라 최적의 구현 방법을 추천할 수 있습니다.

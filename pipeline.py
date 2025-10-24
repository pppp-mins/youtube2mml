"""
YouTube2MML Pipeline
Complete pipeline from YouTube URL to MML code
"""

import os
from pathlib import Path
from typing import Optional, Dict
import time

# Set environment variables for RTX 5090 (Blackwell) compatibility
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from features.youtube_downloader import YouTubeDownloader
from features.audio_preprocessor import AudioPreprocessor
from features.source_separator import SourceSeparator
from features.music_transcriber import MusicTranscriber
from features.music_analyzer import MusicAnalyzer
from features.mml_generator import MMLGenerator


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass


class DownloadError(PipelineError):
    """Failed to download from YouTube"""
    pass


class SeparationError(PipelineError):
    """Failed to separate audio sources"""
    pass


class TranscriptionError(PipelineError):
    """Failed to transcribe audio"""
    pass


class YouTube2MMLPipeline:
    """Complete pipeline from YouTube URL to MML code"""

    def __init__(self, output_dir: str = "downloads"):
        """
        Initialize pipeline

        Args:
            output_dir: Directory to save downloaded audio and MML files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        print("Initializing pipeline components...")
        self.downloader = YouTubeDownloader(output_dir=str(self.output_dir))
        self.preprocessor = AudioPreprocessor(sample_rate=44100)
        self.separator = None  # Will be initialized on first use
        self.transcriber = MusicTranscriber(min_confidence=0.6, min_duration=0.05)
        self.analyzer = MusicAnalyzer(quantize_resolution=16)
        self.generator = MMLGenerator(compact=True)  # Compact mode (no spaces)
        print("Pipeline initialized successfully!")

    def process(self, youtube_url: str, num_harmonies: int = 2) -> Dict:
        """
        Process YouTube URL to MML code

        Args:
            youtube_url: YouTube video URL
            num_harmonies: Number of harmony tracks (1 or 2)

        Returns:
            Dictionary with keys:
                - mml_code: MML code string
                - audio_path: Path to downloaded audio
                - title: Song title
                - tempo: Detected tempo
                - key: Detected key signature
                - processing_time: Total processing time

        Raises:
            DownloadError: If download fails
            SeparationError: If source separation fails
            TranscriptionError: If transcription fails
            PipelineError: If any other step fails
        """
        start_time = time.time()

        try:
            # Stage 1: Download audio
            print("\n" + "=" * 60)
            print("[1/6] Downloading audio from YouTube...")
            print("=" * 60)
            try:
                # TODO: Audio Download
                # audio_path = self.downloader.download_audio(youtube_url)
                audio_path = "downloads/jing rey de los bandidos ost jing girl.mp3"
                title = Path(audio_path).stem
                print(f"✓ Downloaded: {audio_path}")
            except Exception as e:
                raise DownloadError(f"Failed to download: {e}")

            # Stage 2: Preprocess audio
            print("\n" + "=" * 60)
            print("[2/6] Preprocessing audio...")
            print("=" * 60)
            try:
                audio, sr = self.preprocessor.load_and_preprocess(audio_path)
                print(f"✓ Audio loaded: {len(audio)/sr:.2f}s @ {sr} Hz")
            except Exception as e:
                raise PipelineError(f"Failed to preprocess audio: {e}")

            # Stage 3: Separate audio sources
            print("\n" + "=" * 60)
            print("[3/6] Separating audio sources (AI)...")
            print("=" * 60)

            # Initialize separator if not already done
            if self.separator is None:
                self.separator = SourceSeparator(model_name="htdemucs", force_cpu=False)

            try:
                stems = self.separator.separate(audio, sr)
                print(f"✓ Separated into {len(stems)} stems")

                track_audios = self.separator.extract_tracks_for_mml(stems)
                print(f"✓ Extracted {len(track_audios)} tracks for MML")
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a GPU-related error
                if "cuda" in error_str or "gpu" in error_str or "kernel image" in error_str:
                    print(f"\n⚠ GPU error detected: {e}")
                    print("⚠ Retrying with CPU mode (this will be slower)...\n")

                    # Retry with CPU mode
                    try:
                        self.separator = SourceSeparator(model_name="htdemucs", force_cpu=True)
                        stems = self.separator.separate(audio, sr)
                        print(f"✓ Separated into {len(stems)} stems (CPU mode)")

                        track_audios = self.separator.extract_tracks_for_mml(stems)
                        print(f"✓ Extracted {len(track_audios)} tracks for MML")
                    except Exception as e2:
                        raise SeparationError(f"Failed to separate audio even in CPU mode: {e2}")
                else:
                    raise SeparationError(f"Failed to separate audio: {e}")

            # Stage 4: Transcribe music
            print("\n" + "=" * 60)
            print("[4/6] Transcribing music (AI)...")
            print("=" * 60)
            try:
                tracks = []

                # Melody
                print("  Transcribing melody...")
                melody_track = self.transcriber.transcribe_track(
                    track_audios["melody"], sr, "melody"
                )
                tracks.append(melody_track)
                print(f"  ✓ Melody: {len(melody_track.notes)} notes")

                # Harmonies
                for i in range(1, min(num_harmonies, 2) + 1):
                    harmony_key = f"harmony{i}"
                    if harmony_key in track_audios:
                        print(f"  Transcribing harmony {i}...")
                        harmony_track = self.transcriber.transcribe_track(
                            track_audios[harmony_key], sr, harmony_key
                        )
                        tracks.append(harmony_track)
                        print(f"  ✓ Harmony {i}: {len(harmony_track.notes)} notes")

                print(f"✓ Transcribed {len(tracks)} tracks")
            except Exception as e:
                raise TranscriptionError(f"Failed to transcribe audio: {e}")

            # Stage 5: Analyze music
            print("\n" + "=" * 60)
            print("[5/6] Analyzing music (tempo, key, quantization)...")
            print("=" * 60)
            try:
                tracks = self.analyzer.analyze_and_process(audio, sr, tracks)
                tempo = tracks[0].tempo if tracks else 120
                key = tracks[0].key_signature if tracks else "C"
                print(f"✓ Detected tempo: {tempo} BPM")
                print(f"✓ Detected key: {key}")
                print(f"✓ Quantized to 1/16 notes")
            except Exception as e:
                raise PipelineError(f"Failed to analyze music: {e}")

            # Stage 6: Generate MML
            print("\n" + "=" * 60)
            print("[6/6] Generating MML code...")
            print("=" * 60)
            try:
                mml_code = self.generator.generate(tracks, title)
                print(f"✓ Generated MML code ({len(mml_code)} characters)")
            except Exception as e:
                raise PipelineError(f"Failed to generate MML: {e}")

            processing_time = time.time() - start_time

            print("\n" + "=" * 60)
            print(f"✓ COMPLETE! Processing time: {processing_time:.1f}s")
            print("=" * 60)

            return {
                "mml_code": mml_code,
                "audio_path": audio_path,
                "title": title,
                "tempo": tempo,
                "key": key,
                "processing_time": processing_time
            }

        except (DownloadError, SeparationError, TranscriptionError, PipelineError):
            raise
        except Exception as e:
            raise PipelineError(f"Unexpected error in pipeline: {e}")

    def process_and_save(self, youtube_url: str, output_path: Optional[str] = None,
                        num_harmonies: int = 2) -> str:
        """
        Process and save MML to file

        Args:
            youtube_url: YouTube video URL
            output_path: Path to save MML file (optional)
            num_harmonies: Number of harmony tracks (1 or 2)

        Returns:
            Path to saved MML file

        Raises:
            Same as process() method
        """
        result = self.process(youtube_url, num_harmonies)

        if output_path is None:
            output_path = self.output_dir / f"{result['title']}.mml"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['mml_code'])

        print(f"\n✓ MML saved to: {output_path}")
        return str(output_path)


def main():
    """Example usage of YouTube2MMLPipeline"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <youtube_url> [num_harmonies]")
        print("\nExample:")
        print("  python pipeline.py https://youtu.be/nqIxmmPB7eQ 2")
        return

    youtube_url = sys.argv[1]
    num_harmonies = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Initialize pipeline
    pipeline = YouTube2MMLPipeline(output_dir="downloads")

    try:
        # Process YouTube URL
        mml_path = pipeline.process_and_save(youtube_url, num_harmonies=num_harmonies)

        print("\n" + "=" * 60)
        print("MML CODE PREVIEW:")
        print("=" * 60)
        with open(mml_path, 'r', encoding='utf-8') as f:
            print(f.read())

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

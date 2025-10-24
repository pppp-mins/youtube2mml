# Youtube to MML
> It extract audio from a given Youtube URL and generates MML code using fine-tuned and third-party models.
--- ---
![Python](https://img.shields.io/badge/Python-3.11%20|%203.12-blue)
![Documentation](https://img.shields.io/badge/Documentation-DOCS.md-yellow)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-green)

## ✨ Introduction & Structure

**Youtube2MML** converts YouTube audio into MML (Music Macro Language) code, extracting melody and harmony tracks from audio files.

### System Architecture

```
YouTube URL
    ↓
[1] YouTube Downloader (features/youtube_downloader.py)
    ↓
Audio File (MP3/MP4)
    ↓
[2] Audio Preprocessor (features/audio_preprocessor.py)
    ├─ Audio loading & validation
    ├─ Sample rate normalization
    └─ Audio segmentation (if needed)
    ↓
Preprocessed Audio
    ↓
[3] Source Separation (features/source_separator.py)
    ├─ Vocal separation
    ├─ Instrument separation
    └─ Multi-track extraction
    ↓
Separated Tracks (Melody + Harmony 1~N)
    ↓
[4] Music Transcription (features/music_transcriber.py)
    ├─ Pitch detection
    ├─ Onset/offset detection
    ├─ Rhythm analysis
    └─ Note extraction per track
    ↓
Musical Note Data
    ↓
[5] MML Generator (features/mml_generator.py)
    ├─ Note → MML syntax conversion
    ├─ Tempo & time signature setting
    ├─ Track formatting (Melody, Harmony1~N)
    └─ MML code generation
    ↓
MML Code Output
```

### Key Components

1. **YouTube Downloader**: Downloads audio/video from YouTube URLs
2. **Audio Preprocessor**: Prepares audio for analysis (resampling, normalization)
3. **Source Separator**: Separates audio into melody and harmony tracks using AI models
4. **Music Transcriber**: Converts audio signals to musical note data (pitch, duration, timing)
5. **MML Generator**: Converts musical notes into MML syntax format

### Key Features

- Downloads audio from YouTube URLs
- Separates melody and multiple harmony tracks
- Transcribes audio to musical notation
- Generates MML code with melody and harmony tracks
- Supports fine-tuned and third-party AI models


## ⚙️ Installation

### Prerequisites
- **Python**: 3.11+ (Recommended: Python 3.12)
- **ffmpeg**: Required for audio processing and YouTube downloads
- **CUDA** (Optional): For GPU-accelerated source separation

### Installation Steps

Open a terminal and run:

```shell
# Miniconda3
# For Python virtual environments, we recommend miniconda3.
$ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
$ sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3

# Clone project
$ cd ~
$ git clone https://github.com/hli-ai-service/youtube2mml.git

# Python Virtual Environment
# The official supported version is Python 3.11+, other versions have not been tested.
# We recommend Python 3.12.x
$ cd ~/youtube2mml
$ conda create -n youtube2mml python=3.12
$ conda activate youtube2mml

# Install PyTorch with CUDA support first (recommended for GPU acceleration)
$ pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
$ pip install -r requirements.txt

# Check system dependencies
$ chmod +x ./manage.sh
$ ./manage.sh start
```

### Common Installation Issues

#### NumPy Version Compatibility
If you encounter an error about NumPy 2.x compatibility:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.4...
```

**Solution**: This project uses NumPy 1.26.4 (specified in requirements.txt). If you have NumPy 2.x installed:
```bash
pip install "numpy<2"
```

#### PyTorch Installation
- **GPU users**: Install PyTorch with CUDA support BEFORE other dependencies
- **CPU-only users**: Skip the PyTorch installation step; it will be installed automatically with compatible versions

#### ffmpeg Required
Make sure ffmpeg is installed on your system:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 🚀 Usage
```bash
$ ./manage.sh start
```

## 📚 Documentation
[Documents Introduction](docs/DOCS.md)

## ⚖️ License
This project is licensed under the [GNU General Public License v3.0](LICENSE).  
See the LICENSE file for more details.

--- ---
(c) 2025 MinSeok Kim(@pppp-mins). All Rights Reserved.

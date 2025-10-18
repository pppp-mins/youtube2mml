"""
YouTube Media Downloader Module
Downloads audio and video from YouTube using yt-dlp
"""

import yt_dlp
import os
from pathlib import Path
from typing import Optional, Literal


class YouTubeDownloader:
    """
    A clean interface for downloading audio or video from YouTube.

    Usage:
        downloader = YouTubeDownloader(output_dir="downloads")
        # Download audio
        audio_path = downloader.download_audio(url="https://youtube.com/watch?v=...")
        # Download video
        video_path = downloader.download_video(url="https://youtube.com/watch?v=...")
    """

    def __init__(self, output_dir: str = "downloads"):
        """
        Initialize the YouTube downloader.

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_audio(self, url: str, audio_format: str = "mp3",
                       audio_quality: str = "192") -> str:
        """
        Download audio from a YouTube video.

        Args:
            url: YouTube video URL
            audio_format: Output audio format (mp3, m4a, wav, etc.)
            audio_quality: Audio quality in kbps

        Returns:
            Path to the downloaded audio file

        Raises:
            RuntimeError: If download fails
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
                'preferredquality': audio_quality,
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                base_name = os.path.splitext(filename)[0]
                output_file = f"{base_name}.{audio_format}"
                return output_file
        except Exception as e:
            raise RuntimeError(f"Failed to download audio from {url}: {str(e)}")

    def download_video(self, url: str, video_quality: str = "best") -> str:
        """
        Download video from a YouTube video.

        Args:
            url: YouTube video URL
            video_quality: Video quality (best, 1080p, 720p, 480p, etc.)

        Returns:
            Path to the downloaded video file

        Raises:
            RuntimeError: If download fails
        """
        # Format selection based on quality
        if video_quality == "best":
            format_string = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        else:
            # Handle specific resolutions like 1080p, 720p, etc.
            format_string = f'bestvideo[height<={video_quality.replace("p", "")}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

        ydl_opts = {
            'format': format_string,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                base_name = os.path.splitext(filename)[0]
                output_file = f"{base_name}.mp4"
                return output_file
        except Exception as e:
            raise RuntimeError(f"Failed to download video from {url}: {str(e)}")


def main():
    """Example usage of YouTubeDownloader"""
    print("=" * 50)
    print("YouTube Media Downloader")
    print("=" * 50)

    url = input("\nEnter YouTube URL: ").strip()

    if not url:
        print("No URL provided. Exiting.")
        return

    media_type = input("Download type (audio/video) [default: audio]: ").strip().lower()
    if media_type not in ["audio", "video"]:
        media_type = "audio"

    try:
        downloader = YouTubeDownloader(output_dir="downloads")
        print(f"\nDownloading {media_type} from: {url}")

        if media_type == "audio":
            file_path = downloader.download_audio(url)
        else:
            file_path = downloader.download_video(url)

        print(f"Successfully downloaded to: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

"""
Test script for YouTube Downloader
"""

from features.youtube_downloader import YouTubeDownloader


def test_youtube_downloader():
    """Test YouTube audio and video download"""
    test_url = "https://youtu.be/nqIxmmPB7eQ?si=Xk9HkQKmjdqmhGEz"

    print("=" * 60)
    print("YouTube Downloader Test")
    print("=" * 60)
    print(f"Test URL: {test_url}\n")

    # Initialize downloader
    downloader = YouTubeDownloader(output_dir="downloads")

    # Test 1: Download audio (MP3)
    print("\n[Test 1] Downloading audio (MP3)...")
    print("-" * 60)
    try:
        audio_path = downloader.download_audio(test_url)
        print(f" Audio download successful!")
        print(f"  File path: {audio_path}")
    except Exception as e:
        print(f" Audio download failed: {e}")

    # Test 2: Download video (MP4)
    print("\n[Test 2] Downloading video (MP4)...")
    print("-" * 60)
    try:
        video_path = downloader.download_video(test_url)
        print(f" Video download successful!")
        print(f"  File path: {video_path}")
    except Exception as e:
        print(f" Video download failed: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_youtube_downloader()

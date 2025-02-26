# YouTube Audio and Transcript Downloader

This script is designed to download audio and transcripts from a list of YouTube URLs. Each downloaded file is named after its corresponding YouTube video ID, ensuring a clear and consistent naming convention that facilitates easy identification and organization.

## Features

- Downloads the best available audio in WAV format.
- Downloads subtitles (transcripts) in the specified language.
- Saves audio and transcripts with the YouTube video ID as the filename.

## Requirements

- Python 3.6 or higher
- `yt-dlp` library
- `ffmpeg` (required by `yt-dlp` for audio extraction)

## Installation

1. Ensure you have Python 3.6 or higher installed on your system.
2. Install `yt-dlp` using pip:

##Usage
 python .\download.py channel_url output_audio_dir output_subtitle_dir





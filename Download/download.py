import argparse
from downloadutils import *

def main(channel_url, audio_output_folder, subtitle_output_folder):
    video_urls = get_youtube_urls(channel_url)
    download_youtube_audios_and_transcripts(audio_output_folder, subtitle_output_folder, video_urls)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube audios and transcripts.")
    parser.add_argument("channel_url", type=str, help="YouTube channel URL")
    parser.add_argument("audio_output_folder", type=str, help="Folder to save audio files")
    parser.add_argument("subtitle_output_folder", type=str, help="Folder to save subtitle files")
    args = parser.parse_args()

    main(args.channel_url, args.audio_output_folder, args.subtitle_output_folder)

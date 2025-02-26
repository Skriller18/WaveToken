import os
import yt_dlp
import logging
import subprocess
import json

# Setup logging
logging.basicConfig(filename='download.log', filemode='w', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Scraping log')

def get_youtube_urls(channel_url):
    '''
    Extract the video urls of a youtube channel
    params:
    channel_url: str: The URL of the YouTube channel
    '''
    command = ['yt-dlp', '-j', '--flat-playlist', channel_url]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return []
    
    video_urls = []
    for line in result.stdout.strip().split('\n'):
        video_data = json.loads(line)
        video_url = f"https://www.youtube.com/watch?v={video_data['id']}"
        video_urls.append(video_url)
    
    return video_urls

def download_youtube_audios_and_transcripts(output_audio_dir, output_transcript_dir, youtube_links):
    """
    Downloads and extracts audio and transcripts from a list of YouTube URLs and saves them in the specified output directories.
    Each file is named after its YouTube video ID.
    params:
        output_audio_dir : str : The directory where the extracted audio files will be saved.
        output_transcript_dir : str : The directory where the transcript files will be saved.
        youtube_links : list : A list of YouTube URLs to download audio and extract transcripts from.
    """
    # Ensure output directories exist
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_transcript_dir, exist_ok=True)

    for link in youtube_links:
        # Define combined options for downloading audio and subtitles
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  
                'preferredquality': '192', 
            }],
            'outtmpl': {
                'default': os.path.join(output_audio_dir, '%(id)s.%(ext)s'),
                'subtitle': os.path.join(output_transcript_dir, '%(id)s.%(ext)s'),
            },
            'writesubtitles': True,
            'subtitleslangs': ['en-US','en'],  # Change 'en' to your preferred language code
            'writeautomaticsub': True,
        }

        try:
            # Download audio and transcript
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([link])
            
            video_id = link.split('=')[-1]
            logger.info(f"Download completed for audio and transcript with ID {video_id}")
        except Exception as e:
            logger.error(f"Failed to download for {link}. Error: {e}")
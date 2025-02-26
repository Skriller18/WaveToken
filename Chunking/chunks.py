import os
import re
import argparse
import logging
from pydub import AudioSegment

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'chunking.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_vtt(vtt_content):
    captions = []
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\n|\Z)'
    matches = re.findall(pattern, vtt_content, re.DOTALL)
    
    for start, end, text in matches:
        captions.append({
            'start': start,
            'end': end,
            'text': text.strip()
        })
    
    return captions

def time_to_milliseconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

def process_files(subtitle_dir, audio_dir, output_base_dir, logger):
    for subtitle_file in os.listdir(subtitle_dir):
        if subtitle_file.endswith('.hi.vtt'):
            base_name = subtitle_file[:-7]  # Remove '.hi.vtt'
            audio_file = os.path.join(audio_dir, f"{base_name}.wav")
            
            if not os.path.exists(audio_file):
                logger.warning(f"Audio file not found for {subtitle_file}. Skipping.")
                continue
            
            output_dir = os.path.join(output_base_dir, base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Processing {subtitle_file}")
            
            # Process subtitle file
            with open(os.path.join(subtitle_dir, subtitle_file), 'r', encoding='utf-8') as file:
                vtt_content = file.read()
            
            captions = parse_vtt(vtt_content)
            
            # Copy original subtitle file to output directory
            with open(os.path.join(output_dir, subtitle_file), 'w', encoding='utf-8') as file:
                file.write(vtt_content)
            
            # Load audio file
            audio = AudioSegment.from_wav(audio_file)
            
            for i, caption in enumerate(captions):
                # Create text file for each caption
                text_filename = f'chunk_{i+1:04d}.txt'
                with open(os.path.join(output_dir, text_filename), 'w', encoding='utf-8') as file:
                    file.write(f"Start: {caption['start']}\n")
                    file.write(f"End: {caption['end']}\n")
                    file.write(f"Text: {caption['text']}\n")
                
                # Extract audio segment
                start_ms = time_to_milliseconds(caption['start'])
                end_ms = time_to_milliseconds(caption['end'])
                audio_segment = audio[start_ms:end_ms]
                
                # Export audio segment
                audio_filename = f'chunk_{i+1:04d}.wav'
                audio_segment.export(os.path.join(output_dir, audio_filename), format="wav")
                
                logger.debug(f"Created chunk {i+1} for {subtitle_file}")
            
            logger.info(f"Completed processing {subtitle_file}")

def main():
    parser = argparse.ArgumentParser(description='Process subtitle and audio files.')
    parser.add_argument('subtitle_dir', help='Directory containing subtitle files')
    parser.add_argument('audio_dir', help='Directory containing audio files')
    parser.add_argument('output_dir', help='Directory for output')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting processing")
    process_files(args.subtitle_dir, args.audio_dir, args.output_dir, logger)
    logger.info("Processing completed")

if __name__ == "__main__":
    main()

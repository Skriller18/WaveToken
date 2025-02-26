import os
import re
import argparse
import logging
import json
from pydub import AudioSegment

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
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
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n([^-->]*)'
    matches = re.findall(pattern, vtt_content, re.DOTALL)
    
    for start, end, text in matches:
        text = re.sub(r'\s+', ' ', text).strip()
        captions.append({
            'start': start,
            'end': end,
            'text': text
        })
    
    return captions

def time_to_milliseconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

def chunk_based_on_rules(captions):
    cumulative_start = None
    cumulative_text = ""
    intervals = []
    
    i = 0
    while i < len(captions):
        caption = captions[i]
        if not caption['text']:  # Skip empty texts
            i += 1
            continue

        start_ms = time_to_milliseconds(caption['start'])
        end_ms = time_to_milliseconds(caption['end'])
        duration = end_ms - start_ms

        if cumulative_start is None:
            cumulative_start = start_ms
        
        cumulative_text += " " + caption['text']
        cumulative_text = cumulative_text.strip()
        
        if duration > 8000:
            line_end = cumulative_start + 8000
            intervals.append({
                'start': cumulative_start,
                'end': line_end,
                'text': cumulative_text
            })
            cumulative_start = line_end
            cumulative_text = ""
        elif caption['text'][-1] == '.':
            line_end = start_ms + duration
            if line_end - cumulative_start >= 2000:
                intervals.append({
                    'start': cumulative_start,
                    'end': line_end,
                    'text': cumulative_text
                })
                cumulative_start = line_end
                cumulative_text = ""
            else:
                cumulative_text = re.sub(r'[.,]', '', cumulative_text)
        elif caption['text'][-1] == ',':
            # do not create chunk, just accumulate this text with the next one
            i += 1
            continue
        else:
            line_end = start_ms + duration
            if line_end - cumulative_start < 2000:
                i += 1
                continue  # skip this line
            elif duration <= 8000:
                intervals.append({
                    'start': cumulative_start,
                    'end': line_end,
                    'text': cumulative_text
                })
                cumulative_start = line_end
                cumulative_text = ""
                
        i += 1

    if cumulative_text:
        intervals.append({
            'start': cumulative_start,
            'end': time_to_milliseconds(captions[-1]['end']),
            'text': cumulative_text
        })

    return intervals

def process_files(subtitle_dir, audio_dir, output_base_dir, language, subtitles, category, logger, channel):
    jsonl_data = []

    for subtitle_file in os.listdir(subtitle_dir):
        if subtitle_file.endswith('.en.vtt'):
            base_name = subtitle_file[:-7]  # Remove '.hi_final.vtt'
            audio_file = os.path.join(audio_dir, f"{base_name}.wav")
            
            if not os.path.exists(audio_file):
                logger.warning(f"Audio file not found for {subtitle_file}. Skipping.")
                continue
            
            output_dir = os.path.join(output_base_dir, base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            audio_chunks_dir = os.path.join(output_dir, 'audio_chunks')
            subtitle_chunks_dir = os.path.join(output_dir, 'subtitle_chunks')
            os.makedirs(audio_chunks_dir, exist_ok=True)
            os.makedirs(subtitle_chunks_dir, exist_ok=True)
            
            logger.info(f"Processing {subtitle_file}")
            
            # Process subtitle file
            with open(os.path.join(subtitle_dir, subtitle_file), 'r', encoding='utf-8') as file:
                vtt_content = file.read()
            
            captions = parse_vtt(vtt_content)
            
            # Copy original subtitle file to output directory
            with open(os.path.join(subtitle_chunks_dir, subtitle_file), 'w', encoding='utf-8') as file:
                file.write(vtt_content)
            
            # Load audio file
            audio = AudioSegment.from_file(audio_file)
            
            intervals = chunk_based_on_rules(captions)

            for i, interval in enumerate(intervals):
                if interval['end'] - interval['start'] < 2000:
                    continue  # skip this chunk

                chunk_id = f'{channel}_{base_name}_chunk_{interval["start"]}_{interval["end"]}'
                
                # Create text file for each caption chunk
                text_filename = f'{chunk_id}.txt'
                with open(os.path.join(subtitle_chunks_dir, text_filename), 'w', encoding='utf-8') as file:
                    file.write(f'Start: {interval["start"]}\n')
                    file.write(f'End: {interval["end"]}\n')
                    file.write(f'Text: {interval["text"]}\n')

                # Extract and export audio segment
                audio_segment = audio[interval['start']:interval['end']]
                audio_filename = f'{chunk_id}.wav'
                audio_segment.export(os.path.join(audio_chunks_dir, audio_filename), format="wav")

                # Prepare JSONL entry
                jsonl_data.append({
                    "chunk_id": audio_filename,
                    "text": interval['text'],
                    "audio": os.path.join(base_name, 'audio_chunks', audio_filename),
                    "begin_time": interval['start'],
                    "end_time": interval['end'],
                    "audio_id": f'{channel}_{base_name}',
                    "url": "",  # Placeholder if URL is available
                    "source": channel,
                    "language": language,
                    "subtitles": subtitles,
                    "category": category
                })
                
                logger.debug(f"Created chunk {i+1} for {subtitle_file}")
            
            logger.info(f"Completed processing {subtitle_file}")
    
    # Write JSONL file
    jsonl_path = os.path.join(output_base_dir, 'chunks_metadata.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as file:
        for entry in jsonl_data:
            file.write(json.dumps(entry) + '\n')
    
    logger.info(f"JSONL file created at {jsonl_path}")

def main():
    parser = argparse.ArgumentParser(description='Process subtitle and audio files.')
    parser.add_argument('subtitle_dir', help='Directory containing subtitle files')
    parser.add_argument('audio_dir', help='Directory containing audio files')
    parser.add_argument('output_dir', help='Directory for output')
    parser.add_argument('--channel', default='base', help='Channel name')
    parser.add_argument('--language', default='en', help='Language of the subtitles')
    parser.add_argument('--subtitles', default='auto', help='Subtitle source: uploaded/auto')
    parser.add_argument('--category', default='general', help='Category of the content: conversation, topical, etc.')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting processing")
    process_files(args.subtitle_dir, args.audio_dir, args.output_dir, args.language, args.subtitles, args.category, logger, args.channel)
    logger.info("Processing completed")

if __name__ == "__main__":
    main()
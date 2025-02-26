import ijson
import os
import requests
import argparse
import logging

# Configure logging
logging.basicConfig(
    filename='download_audios.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_audio(file_url, file_name, output_folder):
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        file_path = os.path.join(output_folder, file_name)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        logging.info(f"Successfully downloaded: {file_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {file_url}. Error: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download audio files from JSON data.")
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing audio data')
    parser.add_argument('output_folder', type=str, help='Folder to save the downloaded audio files')
    args = parser.parse_args()

    # Ensure the folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Stream and process JSON data incrementally
    try:
        with open(args.json_file, 'r', encoding='utf-8') as file:
            parser = ijson.items(file, 'item')
            for item in parser:
                file_url = item.get("file_url")
                file_name = item.get("file_name")
                if file_url and file_name:
                    download_audio(file_url, file_name, args.output_folder)
    except (UnicodeDecodeError, ijson.common.JSONDecodeError) as e:
        logging.error(f"Error reading JSON file {args.json_file}. Error: {e}")

if __name__ == "__main__":
    main()

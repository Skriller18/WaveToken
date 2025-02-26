import os
import re
import argparse

def process_vtt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Remove the 7th line if it exists
    if len(lines) >= 7:
        del lines[6]  # 0-based index, so 6 is the 7th line

    # Join the lines back into a single string
    content = ''.join(lines)

    # Remove the second line of timestamps
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3} align:start position:0%\n)(?:\n)?(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3} align:start position:0%\n)'
    modified_content = re.sub(pattern, r'\1', content)

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.vtt'):
            file_path = os.path.join(directory_path, filename)
            process_vtt_file(file_path)
            print(f"Processed: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Process VTT files in a directory.')
    parser.add_argument('directory', type=str, help='Path to the directory containing VTT files')
    args = parser.parse_args()

    # Process all VTT files in the specified directory
    process_directory(args.directory)

if __name__ == "__main__":
    main()
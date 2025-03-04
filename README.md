# WaveToken Pipeline

WaveToken is a multimodal dataset pipeline that downloads YouTube audio and subtitles, compresses and cleans the data, generates aligned audio–text chunks with metadata, and tokenizes both modalities. This pipeline is designed for projects involving automatic speech recognition (ASR), speech synthesis, or any application requiring synchronized audio and text tokenization.

## Pipeline Overview

### 1. Download
Downloads audio and subtitles from a given YouTube channel.

### 2. Compress
Compresses downloaded audio files to a lower bitrate (64k) using ffmpeg.

### 3. Subtitles Clean
Processes and cleans subtitle (.vtt) files by removing unnecessary lines and duplicate timestamps.

### 4. Generate Chunks
Chunks audio and subtitle files based on pre-defined rules, creates corresponding audio segments, and generates a chunks_metadata.jsonl file.

### 5. Prepare Tokens
Tokenizes audio using a pretrained Mimi model and text using a GPT2-based tokenizer. The tokens are saved as .npy files.


## Prerequisites
    Python 3.7+
    ffmpeg
        - On Ubuntu/Debian: sudo apt-get install ffmpeg
        - On macOS (via Homebrew): brew install ffmpeg
    
    Python Libraries (Install required Python packages if you don’t have a requirements.txt, you can install these packages manually):
        - tqdm
        - pydub
        - torchaudio
        - numpy
        - torch
        - transformers
        - argparse


```
pip install tqdm pydub torchaudio numpy torch transformers
```
## Installation
### Clone the repository and install dependencies:
```
git clone https://github.com/Skriller18/WaveToken.git
cd WaveToken
pip install -r requirements.txt
```
(If you do not have a requirements.txt file, refer to the prerequisites above.)

### Usage

1.***Download Audio and Subtitles***

Run the download.py script to download audio and subtitle files from a specified YouTube channel:

```
python download.py <channel_url> <audio_output_folder> <subtitle_output_folder>
```

2.***Compress Audio Files***

Use compress.py to compress downloaded audio files to a lower bitrate. This script searches for audio files in the input directory and compresses them:
```
python compress.py --inp <audio_input_directory> --out <compressed_output_directory>
```

3.***Clean Subtitle Files***

Clean and process the subtitle (.vtt) files using sub_mod.py:
```
python sub_mod.py <subtitle_directory>
```

4.***Generate Chunks and Metadata***

Chunk the subtitles and corresponding audio files and generate a metadata JSONL file using jsonl.gen_final.py. This script also extracts audio segments and creates text files for each chunk.
```
python jsonl.gen_final.py <subtitle_dir> <audio_dir> <output_dir> --channel <channel_name> --language <language> --subtitles <subtitle_source> --category <category>
```

5.***Prepare and Save Tokens***

Tokenize the generated chunks using prepare_tokens.py. This step processes the JSONL metadata file, loads audio and text, and generates token files saved as .npy.
```
python prepare_tokens.py --jsonl <jsonl_file_path> --dataset_dir <dataset_directory> --out_dir <output_tokens_directory> --device <device>
```
Example:

## Notes
#### Directory Structure
Ensure that the directory structure is correctly set up. Each script will create necessary subdirectories (e.g., for compressed files, chunk outputs, token outputs).

### Logging
The jsonl.gen_final.py script logs its progress and errors to chunking.log in the output directory.

### Error Handling
If any audio file is missing or a subtitle file is not properly formatted, the scripts will log warnings and continue processing.

### Troubleshooting
- #### ffmpeg Not Found:
    Ensure ffmpeg is installed and accessible in your PATH.

- #### File Paths:
    Double-check that input and output paths are correct, especially on Windows where backslashes might be an issue. The scripts normalize paths accordingly.

- #### Device Issues:
    If you encounter CUDA errors, make sure you have the appropriate CUDA drivers installed, or switch to CPU mode by specifying --device cpu.

## License

MIT License

Copyright (c) 2023 WaveToken Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
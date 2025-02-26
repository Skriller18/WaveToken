import os
import json
import argparse
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from transformers import MimiModel, AutoFeatureExtractor, AutoTokenizer

class TextTokenizer:
    def __init__(self, name='cmeraki/gpt2-124M-400B'):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        print("Text vocab size:", self.tokenizer.vocab_size)

    def encode(self, text: str):
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class MimiTokenizer:
    def __init__(self, device):
        self.device = device
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.model.to(device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi", device=device)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.n_codebooks = 8

    @torch.inference_mode()
    def encode(self, waveform):
        inputs = self.feature_extractor(
            raw_audio=waveform, 
            sampling_rate=self.sampling_rate, 
            return_tensors="pt"
        ).to(self.device)

        output = self.model.encode(
            inputs["input_values"], 
            inputs["padding_mask"], 
            num_quantizers=self.n_codebooks
        )
        tokens = output.audio_codes[0].cpu().numpy()
        return tokens

    def decode(self, tokens):
        assert len(tokens.shape) == 2
        tokens = torch.tensor(np.expand_dims(tokens, axis=0)).to(self.device)
        output = self.model.decode(tokens)
        waveform = output.audio_values.cpu()
        return waveform


def process_jsonl(jsonl_path: str, dataset_dir: str, out_dir: str, device: str = "cuda:0"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Initialize tokenizers
    audio_tokenizer = MimiTokenizer(device=device)
    text_tokenizer = TextTokenizer()

    # Read JSONL file
    with open(jsonl_path, "r") as file:
        lines = file.readlines()
    
    for line in tqdm(lines, desc="Processing chunks"):
        chunk = json.loads(line)
        # Ensure chunk_id exists
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("Missing chunk_id in JSONL.")
        
        # Get audio file path and normalize path (handle Windows backslashes)
        chunk["audio"] = chunk["audio"].replace("\\", "/")
        audio_path = os.path.join(dataset_dir, chunk["audio"])
        text = chunk["text"]

        # Load the audio file
        waveform, sr = torchaudio.load(audio_path)
        # Resample if needed
        if sr != audio_tokenizer.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=audio_tokenizer.sampling_rate)
            waveform = resampler(waveform)

        # Assume single channel
        waveform_np = waveform.squeeze().numpy()
        # Tokenize audio and text
        audio_tokens = audio_tokenizer.encode(waveform_np)
        text_tokens = text_tokenizer.encode(text)

        # Save tokens as .npy files
        audio_filename = os.path.join(out_dir, f"{chunk_id}_audio.npy")
        text_filename = os.path.join(out_dir, f"{chunk_id}_text.npy")
        np.save(audio_filename, audio_tokens)
        np.save(text_filename, np.array(text_tokens))

    print(f"Tokenization complete! Files saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize audio and text from a JSONL file into .npy files.")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to the JSONL file")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to store output .npy files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (eg, 'cuda:0' or 'cpu')")
    args = parser.parse_args()
    process_jsonl(args.jsonl, args.dataset_dir, args.out_dir, args.device)
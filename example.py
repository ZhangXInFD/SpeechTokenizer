import argparse
import torchaudio
import torch
from speechtokenizer import SpeechTokenizer
from scipy.io.wavfile import write
import numpy as np

from huggingface_hub import snapshot_download

snapshot_download(repo_id="fnlp/SpeechTokenizer", local_dir="model_hub")


# Set up argument parser
parser = argparse.ArgumentParser(
    description="Load SpeechTokenizer model and process audio file."
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the model configuration file.",
    default="model_hub/speechtokenizer_hubert_avg/config.json",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    help="Path to the model checkpoint file.",
    default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
)
parser.add_argument(
    "--speech_file",
    type=str,
    required=True,
    help="Path to the speech file to be processed.",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="Path to save the output audio file.",
    default="example_output.wav",
)

args = parser.parse_args()

# Load model from the specified checkpoint
model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
model.eval()

# Determine the model's expected sample rate
model_sample_rate = model.sample_rate

# Load and preprocess speech waveform with the model's sample rate
wav, sr = torchaudio.load(args.speech_file)

if sr != model_sample_rate:
    resample_transform = torchaudio.transforms.Resample(
        orig_freq=sr, new_freq=model_sample_rate
    )
    wav = resample_transform(wav)

# Ensure the waveform is monophonic
if wav.shape[0] > 1:
    wav = wav[:1, :]

wav = wav.unsqueeze(0)


# Extract discrete codes from SpeechTokenizer
with torch.no_grad():
    codes = model.encode(wav)  # codes: (n_q, B, T)

RVQ_1 = codes[:1, :, :]  # Contain content info, can be considered as semantic tokens
RVQ_supplement = codes[
    1:, :, :
]  # Contain timbre info, complete info lost by the first quantizer

# Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
wav_out = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))

# Decoding from RVQ-i:j tokens from the ith quantizers to the jth quantizers
# Example: decoding from quantizer 0 to quantizer 2
wav_out = wav_out.detach().numpy()
write(args.output_file, model_sample_rate, wav_out.astype(np.float32))

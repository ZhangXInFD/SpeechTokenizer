# SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models

<a href='https://0nutation.github.io/SpeechTokenizer.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Introduction
This is the code for the SpeechTokenizer presented in the [SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models](https://0nutation.github.io/SpeechTokenizer.github.io/). SpeechTokenizer is a unified speech tokenizer for speech large language models, which adopts the Encoder-Decoder architecture with residual vector quantization (RVQ). Unifying semantic and acoustic tokens, SpeechTokenizer disentangles different aspects of speech information hierarchically across different RVQ layers. Specifically, The code indices that the first quantizer of RVQ outputs can be considered as semantic tokens and the output of the remaining quantizers can be regarded as acoustic tokens, which serve as supplements for the information lost by the first quantizer. We provide our models:
* A model operated at 16khz on monophonic speech trained on Librispeech with average representation across all HuBERT layers as semantic teacher.

<br>
<p align="center">
    <img src="images/overview.png" width="95%"> <br>
    Overview
</p>
<p align="center">
    <img src="images/speechtokenizer_framework.jpg" width="95%"> <br>
    The SpeechTokenizer framework.
</p>
<br>


Welcome to try our [SLMTokBench](https://github.com/0nutation/SLMTokBench) 
 and we will also open source our  [USLM](https://github.com/0nutation/USLM) !!



## Samples

Samples are provided on [our demo page](https://0nutation.github.io/SpeechTokenizer.github.io/).

## Installation

SpeechTokenizer requires Python>=3.8, and a reasonly recent version of PyTorch.
To install SpeechTokenizer, you can run from this repository:
```bash
# pip install -U speechtokenizer
git clone https://github.com/ZhangXInFD/SpeechTokenizer.git
cd SpeechTokenizer
pip install .
```
## Usage
### Model storage
[model list]()
### load model
```python
from speechtokenizer import SpeechTokenizer

config_path = '/path/config.json'
ckpt_path = '/path/SpeechTokenizer.pt'
model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
model.eval()
```
### Extracting discrete representions
```python
import torchaudio
import torch

# Load and pre-process speech waveform
wav, sr = torchaudio.load('<SPEECH_FILE_PATH>')
if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
wav = wav.unsqueeze(0)

# Extract discrete codes from SpeechTokenizer
with torch.no_grad():
    codes = model.encode(wav) # codes: (n_q, B, T)

semantic_tokens = codes[0, :, :]
acoustic_tokens = codes[1:, :, :]
```

### Decoding discrete representions
```python
# Decoding from the first quantizers to ith quantizers
wav = model.decode(codes[:(i + 1)]) # wav: (B, 1, T)

# Decoding from ith quantizers to jth quantizers
wav = model.decode(codes[i: (j + 1)], st=i) 

# Cancatenating semantic tokens and acoustic tokens and then decoding
semantic_tokens = ... # (..., B, T)
acoustic_tokens = ... # (..., B, T)
wav = model.decode(torch.cat([semantic_tokens, acoustic_tokens], axis=0))
```

## Citation
If you use this code or result in your paper, please cite our work as:

## License
The code in this repository is released under the Apache 2.0 license as found in the
[LICENSE](LICENSE) file.

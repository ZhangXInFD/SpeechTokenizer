# SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models

<a href='https://0nutation.github.io/SpeechTokenizer.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2308.16692'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

## Introduction
This is the code for the SpeechTokenizer presented in the [SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models](https://arxiv.org/abs/2308.16692). SpeechTokenizer is a unified speech tokenizer for speech language models, which adopts the Encoder-Decoder architecture with residual vector quantization (RVQ). Unifying semantic and acoustic tokens, SpeechTokenizer disentangles different aspects of speech information hierarchically across different RVQ layers. Specifically, the code indices that the first quantizer of RVQ outputs can be considered as semantic tokens and the output of the remaining quantizers mainly contain timbre info, which serve as supplements for the information lost by the first quantizer. We provide our models:
* A model operated at 16khz on monophonic speech trained on Librispeech with average representation across all HuBERT layers as semantic teacher.
* A model with  [Snake activation](https://arxiv.org/abs/2306.06546) operated at 16khz on monophonic speech trained on Librispeech and Common Voice with average representation across all HuBERT layers as semantic teacher.

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
 and we will also open source our [USLM](https://github.com/0nutation/USLM)!

## Qick Link
* [Relase](#release)
* [Samples](#samples)
* [Installation](#installation)
* [Model List](#model-list)
* [Usage](#usage)
* [Train SpeechTokenizer](#train-speechtokenizer)
    * [Data Preprocess](#data-preprocess)
    * [Train](#train)
    * [Quick Start](#quick-start)
* [Citation](#citation)
* [License](#license)


## Release
- [2024/6/9] 🔥 We released the training code of SpeechTokenizer.
- [2024/3] 🔥 We released a checkpoint of SpeechTokenizer with [Snake activation](https://arxiv.org/abs/2306.06546) trained on LibriSpeech and Common Voice.
- [2023/9/11] 🔥 We released code of [soundstorm_speechtokenizer](https://github.com/ZhangXInFD/soundstorm-speechtokenizer).
- [2023/9/10] 🔥 We released code and checkpoints of [USLM](https://github.com/0nutation/USLM). 
- [2023/9/1] 🔥 We released code and checkpoints of SpeechTokenizer. Checkout the [paper](https://arxiv.org/abs/2308.16692) and [demo](https://0nutation.github.io/SpeechTokenizer.github.io/).

## Samples

Samples are provided on [our demo page](https://0nutation.github.io/SpeechTokenizer.github.io/).

## Installation

SpeechTokenizer requires Python>=3.8, and a reasonly recent version of PyTorch.
To install SpeechTokenizer, you can run from this repository:
```bash
pip install -U speechtokenizer

# or you can clone the repo and install locally
git clone https://github.com/ZhangXInFD/SpeechTokenizer.git
cd SpeechTokenizer
pip install .
```
## Model List
| Model| Dataset |Discription|
|:----|:----:|:----|
|[speechtokenizer_hubert_avg](https://huggingface.co/fnlp/SpeechTokenizer/tree/main/speechtokenizer_hubert_avg)|LibriSpeech|Adopt average representation across all HuBERT layers as semantic teacher |
|[speechtokenizer_snake](https://huggingface.co/fnlp/AnyGPT-speech-modules/tree/main/speechtokenizer)|LibriSpeech + Common Voice|Snake activation, average representation across all HuBERT layers |
## Usage
### load model
```python
from speechtokenizer import SpeechTokenizer

config_path = '/path/config.json'
ckpt_path = '/path/SpeechTokenizer.pt'
model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
model.eval()
```
### Extracting discrete representations
```python
import torchaudio
import torch

# Load and pre-process speech waveform
wav, sr = torchaudio.load('<SPEECH_FILE_PATH>')

# monophonic checking
if wav.shape(0) > 1:
    wav = wav[:1,:]

if sr != model.sample_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)

wav = wav.unsqueeze(0)

# Extract discrete codes from SpeechTokenizer
with torch.no_grad():
    codes = model.encode(wav) # codes: (n_q, B, T)

RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer
```

### Decoding discrete representations
```python
# Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
wav = model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))

# Decoding from RVQ-i:j tokens from the ith quantizers to the jth quantizers
wav = model.decode(codes[i: (j + 1)], st=i) 
```

## Train SpeechTokenizer
In the following section, we describe how to train a SpeechTokenizer model by using our trainer.
### Data Preprocess
To train the SpeechTokenizer, the first step is to extract semantic teacher representations from raw audio waveforms. We provide an example of how to extract HuBERT representations in [scripts/hubert_rep_extract.sh](scripts/hubert_rep_extract.sh). We explain the arguments in the following:
* `--config`: Config file path. An example is provided in [config/spt_base_cfg.json](config/spt_base_cfg.json). You can modify the `semantic_model_path` and `semantic_model_layer` parameters in this file to change the Hubert model and the target layer.
* `--audio_dir`: The path to the folder containing all audio files.
* `--rep_dir`: The path to the folder storing all semantic representation files.
* `--exts`: The file extension of the audio files. Use ',' to separate multiple extensions if they exist.
* `--split_seed`: Random seed for splitting training set and validation set.
* `--valid_set_size`: The size of validation set. When this number is between 0 and 1, it represents the proportion of the total dataset used for the validation set.

### Train
You can use SpeechTokenizerTrainer to train a SpeechTokenizer as follows:
```python
from speechtokenizer import SpeechTokenizer, SpeechTokenizerTrainer
from speechtokenizer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json


# Load model and trainer config
with open('<CONFIG_FILE_PATH>') as f:
    cfg = json.load(f)

# Initialize SpeechTokenizer
generator = SpeechTokenizer(cfg)

# Initialize the discriminators. You can add any discriminator that is not yet implemented in this repository, as long as the output format remains consistent with the discriminators in `speechtokenizer.discriminators`.
discriminators = {'mpd':MultiPeriodDiscriminator(), 'msd':MultiScaleDiscriminator(), 'mstftd':MultiScaleSTFTDiscriminator(32)}

# Initialize Trainer
trainer = SpeechTokenizerTrainer(generator=generator,
                                discriminators=discriminators,
                                cfg=cfg)

# Start training
trainer.train()

# Continue training from checkpoints
trainer.continue_train()
```
We provide example training scripts in [scripts/train_example.sh](scripts/train_example.sh). All arguments for SpeechTokenizerTrainer are defined in [config/spt_base.json](config/spt_base.json). Below, we explain some of the important arguments:
* `train_files` and `valid_files`: Training file path and validation file path. These files should be text files listing the paths of all audio files and their corresponding semantic representation files in the training/validation set. Each line should follow the format: "<audio_file_path>\t<semantic_file_path>". If you use [scripts/hubert_rep_extract.sh](scripts/hubert_rep_extract.sh) to extract semantic representations, these two files will be genrated automantically.
* `distill_type`: Use "d_axis" for D-axis distillation loss and "t_axis" for T-axis distillation loss, as mentioned in the paper.

### Quick Start
If you want to fully follow our experimental setup, simply set `semantic_model_path` in [config/spt_base.json](config/spt_base.json), and `AUDIO_DIR`, `REP_DIR`, `EXTS` in [scripts/hubert_rep_extract.sh](scripts/hubert_rep_extract.sh), and other optional arguments , then execute the following code:
```shell
cd SpeechTokenizer

# Extact semantic representation
bash scripts/hubert_rep_extract.sh

# Train
bash scripts/train_example.sh
```
## Citation
If you use this code or result in your paper, please cite our work as:
```Tex
@misc{zhang2023speechtokenizer,
      title={SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models}, 
      author={Xin Zhang and Dong Zhang and Shimin Li and Yaqian Zhou and Xipeng Qiu},
      year={2023},
      eprint={2308.16692},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## License
The code in this repository is released under the Apache 2.0 license as found in the
[LICENSE](LICENSE) file.

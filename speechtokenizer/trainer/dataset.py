from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import torch
import numpy as np

def collate_fn(data):
    # return pad_sequence(data, batch_first=True)
    # return pad_sequence(*data)
    is_one_data = not isinstance(data[0], tuple)
    outputs = []
    if is_one_data:
        for datum in data:
            if isinstance(datum, torch.Tensor):
                output = datum.unsqueeze(0)
            else:
                output = torch.tensor([datum])
            outputs.append(output)
        return tuple(outputs)        
    for datum in zip(*data):
        if isinstance(datum[0], torch.Tensor):
            output = pad_sequence(datum, batch_first=True)
        else:
            output = torch.tensor(list(datum))
        outputs.append(output)

    return tuple(outputs)

def get_dataloader(ds, **kwargs):
    return DataLoader(ds, collate_fn=collate_fn, **kwargs)

class audioDataset(Dataset):
    
    def __init__(self,
                 file_list,
                 segment_size,
                 sample_rate,
                 downsample_rate = 320,
                 valid=False):
        super().__init__()
        self.file_list = file_list
        self.segment_size = segment_size
        self.sample_rate = sample_rate
        self.valid = valid
        self.downsample_rate = downsample_rate
        
    def __len__(self):
        return len(self.file_list)
    
    
    def __getitem__(self, index):
        file = self.file_list[index].strip()
        audio_file, feature_file = file.split('\t')
        audio, sr = torchaudio.load(audio_file)
        feature = torch.from_numpy(np.load(feature_file))
        audio = audio.mean(axis=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        if audio.size(-1) > self.segment_size:
            if self.valid:
                return audio[:self.segment_size], feature[:self.segment_size // self.downsample_rate]
            max_audio_start = audio.size(-1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_size]
            feature_start = min(int(audio_start / self.downsample_rate), feature.size(0) - self.segment_size // self.downsample_rate)
            feature = feature[feature_start:feature_start + self.segment_size // self.downsample_rate, :]
        else:
            if not self.valid:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')
        return audio, feature
from transformers import HubertModel,  Wav2Vec2FeatureExtractor
from pathlib import Path
import torchaudio
import torch
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--audio_dir', type=str, help='Audio folder path')
    parser.add_argument('--rep_dir', type=str, help='Path to save representation files')
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='flac')
    parser.add_argument('--split_seed', type=int, help="Random seed", default=0)
    parser.add_argument('--valid_set_size', type=float, default=1000)
    args = parser.parse_args()
    exts = args.exts.split(',')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.config) as f:
        cfg = json.load(f)
    sample_rate = cfg.get('sample_rate')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.get('semantic_model_path'))
    model = HubertModel.from_pretrained(cfg.get('semantic_model_path')).eval().to(device)
    target_layer = cfg.get('semantic_model_layer')
    path = Path(args.audio_dir)
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)
    train_file_list = cfg.get('train_files')
    valid_file_list = cfg.get('valid_files')
    segment_size = cfg.get('segment_size')
    random.seed(args.split_seed)
    random.shuffle(file_list)
    print(f'A total of {len(file_list)} samples will be processed, and {valid_set_size} of them will be included in the validation set.')
    with torch.no_grad():
        for i, audio_file in tqdm(enumerate(file_list)):
            wav, sr = torchaudio.load(audio_file)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            if wav.size(-1) < segment_size:
                wav = torch.nn.functional.pad(wav, (0, segment_size - wav.size(-1)), 'constant')
            input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
            ouput = model(input_values.to(model.device), output_hidden_states=True)
            if target_layer == 'avg':
                rep = torch.mean(torch.stack(ouput.hidden_states), axis=0)
            else:
                rep = ouput.hidden_states[target_layer]
            rep_file = audio_file.replace(args.audio_dir, args.rep_dir).split('.')[0] + '.hubert.npy'
            rep_sub_dir = '/'.join(rep_file.split('/')[:-1])
            if not os.path.exists(rep_sub_dir):
                os.makedirs(rep_sub_dir)
            np.save(rep_file, rep.detach().cpu().numpy())
            if i < valid_set_size:
                with open(valid_file_list, 'a+') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
            else:
                with open(train_file_list, 'a+') as f:
                    f.write(f'{audio_file}\t{rep_file}\n')
            
            


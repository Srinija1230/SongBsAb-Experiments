

import torch
import yaml
import os
import torchaudio
import numpy as np

from speaker_encoder.resnet import resnet

from pathlib import Path

device = 'cuda'
_device = device

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--attack_flag", type=str, required=True)
parser.add_argument("--attack_flag_2", type=str, required=True)

parser.add_argument('-limit_target_spk', action='store_false', default=True)
parser.add_argument('-limit_source_voice', action='store_false', default=True)
parser.add_argument('-num_source_voice', type=int, default=None)

parser.add_argument('--src_root', '-src_root', type=str, default=None)
parser.add_argument('--des_root', '-des_root', type=str, default=None)

parser.add_argument('-dataset', '--dataset', type=str, default='opensinger', choices=['opensinger'])

subparser = parser.add_subparsers(dest='system_type')

resnet18_veri_parser = subparser.add_parser("resnet18_veri")
resnet18_veri_parser.add_argument('-extractor',
            default='pre-trained-models/auto_speech/autoSpeech-resnet18-veri.pth')
resnet18_veri_parser.add_argument('-mean', default='pre-trained-models/auto_speech/resnet18_veri-emb-mean.pickle')
resnet18_veri_parser.add_argument('-feat_mean',
            default='pre-trained-models/auto_speech/resNet18-feat-mean')
resnet18_veri_parser.add_argument('-feat_std',
            default='pre-trained-models/auto_speech/resNet18-feat-std')
resnet18_veri_parser.add_argument('-model_file', default=None)

args = parser.parse_args()

model_dir_1 = 'saved_models'
model_dir_2 = 'pre-trained-models'
args.dataset_2 = 'vox2'
args.model = 'target'

system_flag = f'-{args.system_type}' if args.system_type != 'lora_LSTM' else ''
args.threshold = -np.infty
if 'resnet' in args.system_type:
    MODEL_NAME = args.system_type.split('_')[0]
    base_model = resnet(MODEL_NAME, args.extractor, mean_file=args.mean, 
                        feat_mean_file=args.feat_mean, feat_std_file=args.feat_std, device=device, model_file=args.model_file, threshold=args.threshold)
else:
    raise NotImplementedError('Unsupported System Type')
speaker_encoder = base_model


data_svc = './storage/data_svc-all_singers-10_voices'
open_singer_root_ori = './storage/OpenSinger'
new_data_svc = './storage/data_svc-{}'.format(args.attack_flag)
if args.src_root is not None:
    data_svc = args.src_root
if args.des_root is not None:
    new_data_svc = args.des_root

all_spk_keys = sorted(os.listdir(data_svc))
des_file = 'select-target_speakers-source_speeches-des.yaml'
with open(des_file, 'r') as f:
    spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
    print(len(spks_keys))
with open(des_file, 'r') as f:
    spk_2_utt = yaml.load(f, Loader=yaml.FullLoader)
if args.limit_target_spk:
    spks_keys = [x for x in all_spk_keys if x in spks_keys]
else:
    spks_keys = all_spk_keys
print(spks_keys, len(spks_keys))

os.makedirs(f'./txt_files/{args.system_type}', exist_ok=True)
normal_self_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag}-self_sim-all_speakers-10_voices.txt'
normal_out_sim_file = f'./txt_files/{args.system_type}/{args.attack_flag_2}-out_sim-all_speakers-10_voices-1000_sources.txt'
w3 = open(normal_out_sim_file, 'w')
all_self_sims = []
all_out_sims = []
with torch.no_grad():
    for idx_spk, flag in enumerate(spks_keys):

        # my mean emb
        wav_dir = os.path.join(data_svc, flag, 'waves')
        if not os.path.exists(wav_dir):
            continue
        print('w0:', wav_dir)
        wav_names = sorted(os.listdir(wav_dir))
        subfile_num = 0
        speaker_ave = 0
        for name in wav_names[:10]:
            if 'wav' not in name:
                continue
            wav_path = os.path.join(wav_dir, name)
            wav, fs = torchaudio.load(wav_path)
            wav = wav.cuda()
            source_embed = speaker_encoder.embedding(wav.unsqueeze(0))
            speaker_ave = speaker_ave + source_embed
            subfile_num = subfile_num + 1
        speaker_ave = speaker_ave / subfile_num

        # adver case out voice sims
        out_voices_sims = []
        des_root = f'./storage/model_pretrain/{flag}/inference-{args.attack_flag_2}'
        print('w3:', des_root)
        if not os.path.exists(des_root):
            continue
        if len(os.listdir(des_root)) <= 0:
            continue
        for x in os.listdir(des_root):
            if 'wav' not in x:
                continue
            if 'pitch.wav' in x:
                continue
            if args.limit_source_voice and x not in spk_2_utt[flag]:
                continue
            path = os.path.join(des_root, x)
            wav, fs = torchaudio.load(path)
            wav = wav.cuda()
            sims = speaker_encoder(wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
            out_voices_sims += sims
            w3.write(f'{flag} {x} {sims[0]}\n')
        print('*' * 5, flag, np.mean(out_voices_sims), np.std(out_voices_sims), np.max(out_voices_sims), np.min(out_voices_sims), '*' * 5)
        all_out_sims += out_voices_sims

w3.close()
print('*' * 10, 'all out sims:', np.mean(all_out_sims), np.std(all_out_sims), np.max(all_out_sims), np.min(all_out_sims), '*' * 10)

import torch
import yaml
import os
import torchaudio
import numpy as np

from speaker_encoder.resnet import resnet

device = 'cuda'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--epsilon", type=float, default=0.02)
parser.add_argument("--max_iter", type=int, default=50)
parser.add_argument("--voice_num", type=int, default=10)

parser.add_argument('-limit_target_spk', action='store_false', default=True)
parser.add_argument('-limit_source_voice', action='store_false', default=True)
parser.add_argument('-num_source_voice', type=int, default=None)

parser.add_argument("--attack_flag", type=str, default=None)
parser.add_argument("--attack_flag_2", type=str, default=None)

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

system_flag = f'-{args.system_type}' if args.system_type != 'lora_LSTM' else ''
args.threshold = -np.infty
if 'resnet' in args.system_type:
    MODEL_NAME = args.system_type.split('_')[0]
    base_model = resnet(MODEL_NAME, args.extractor, mean_file=args.mean, 
                        feat_mean_file=args.feat_mean, feat_std_file=args.feat_std, device=device, model_file=args.model_file, threshold=args.threshold)
else:
    raise NotImplementedError('Unsupported System Type')
speaker_encoder = base_model


data_svc = './storage/data_svc' if args.attack_flag is None else f'./storage/data_svc-{args.attack_flag}'
open_singer_root_ori = './storage/OpenSinger'
print(data_svc)

all_spk_keys = os.listdir(data_svc)
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
source_sim_file = f'./txt_files/{args.system_type}/source_sim.txt'
normal_out_sim_file = f'./txt_files/{args.system_type}/normal-out_sim.txt'

w2 = open(source_sim_file, 'w')
w3 = open(normal_out_sim_file, 'w')
all_self_sims = []
all_source_sims = []
all_out_sims = []
with torch.no_grad():
    for idx_spk, flag in enumerate(spks_keys):
        # my mean emb
        wav_dir = os.path.join(data_svc, flag, 'waves')
        wav_names = sorted(os.listdir(wav_dir))
        subfile_num = 0
        speaker_ave = 0
        all_my_embs = []
        all_my_wavs = []
        for name in wav_names[:10]:
            if 'wav' not in name:
                continue
            wav_path = os.path.join(wav_dir, name)
            wav, fs = torchaudio.load(wav_path)
            wav = wav.cuda()
            source_embed = speaker_encoder.embedding(wav.unsqueeze(0))
            speaker_ave = speaker_ave + source_embed
            subfile_num = subfile_num + 1
            all_my_embs.append(source_embed)
            all_my_wavs.append((name, wav))
        speaker_ave = speaker_ave / subfile_num
        all_my_embs = torch.cat(all_my_embs, dim=0)

        pre_dir = 'model_pretrain'
        des_root = f'./storage/{pre_dir}/{flag}/inference' if args.attack_flag_2 is None \
            else f'./storage/{pre_dir}/{flag}/inference-{args.attack_flag_2}'
        # source voice sim
        source_voices_sims = []
        for g in ['WomanRaw', 'ManRaw']:
            g_dir = os.path.join(open_singer_root_ori, g)
            for x in os.listdir(g_dir):
                if len(x.split('_')) != 2:
                    continue
                flag_2 = '{}_{}'.format('M' if g == 'ManRaw' else 'F', x.split('_')[0])
                gender_flag = 'M' if g == 'ManRaw' else 'F'
                if flag_2 == flag:
                    continue
                cnt = 0
                for name in os.listdir(os.path.join(g_dir, x)):
                    if 'wav' not in name:
                        continue
                    if args.limit_source_voice and gender_flag + '_' + name not in spk_2_utt[flag]:
                        continue
                    if args.num_source_voice is not None and gender_flag + '_' + name not in os.listdir(des_root):
                        continue
                    src_path = os.path.join(g_dir, x, name)
                    src_wav, fs = torchaudio.load(src_path)
                    src_wav = src_wav.cuda()
                    sims = speaker_encoder(src_wav.unsqueeze(0), enroll_embs=speaker_ave).detach().cpu().numpy().flatten().tolist()
                    source_voices_sims += sims
                    w2.write(f'{flag} {gender_flag + "_" + name} {sims[0]}\n')
        print('*' * 5, flag, np.mean(source_voices_sims), np.std(source_voices_sims), np.max(source_voices_sims), np.min(source_voices_sims), '*' * 5)
        all_source_sims += source_voices_sims
            
        # normal case out voice sims
        out_voices_sims = []
        pre_dir = 'model_pretrain'
        des_root = f'./storage/{pre_dir}/{flag}/inference' if args.attack_flag_2 is None \
            else f'./storage/{pre_dir}/{flag}/inference-{args.attack_flag_2}'
        for x in os.listdir(des_root):
            if 'wav' not in x:
                continue
            if 'pitch.wav' in x:
                continue
            # if x not in spk_2_utt[flag]:
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

w2.close()
w3.close()
print('*' * 10, 'all source sims:', np.mean(all_source_sims), np.std(all_source_sims), np.max(all_source_sims), np.min(all_source_sims), '*' * 10)
print('*' * 10, 'all out sims:', np.mean(all_out_sims), np.std(all_out_sims), np.max(all_out_sims), np.min(all_out_sims), '*' * 10)
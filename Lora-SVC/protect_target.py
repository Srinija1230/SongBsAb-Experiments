
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
import os
import copy
import librosa
from pathlib import Path

from masker import Masker
from speaker_encoder.lora_LSTM import LoraLSTM
from content_encoder.whisperppg import WhisperPPG


device = 'cuda'
_device = device


if __name__ == '__main__':

    # # ## $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", "-start", type=int, default=0)
    parser.add_argument("--end", "-end", type=int, default=-1)
    parser.add_argument("--epsilon", "-epsilon", type=float, default=0.03)
    parser.add_argument("--max_iter", "-max_iter", type=int, default=2000)
    parser.add_argument("--lr", '-lr', type=float, default=0.0002)

    parser.add_argument('--backtrack', '-backtrack', action='store_false', default=True)
    parser.add_argument('--src_flag', '-src_root', type=str, default=None)
    parser.add_argument('--des_flag', '-des_root', type=str, default=None)

    parser.add_argument("--voice_num", "-voice_num", type=int, default=10)
    parser.add_argument('--limit_target_spk', '-limit_target_spk', action='store_false', default=True)
    
    subparser = parser.add_subparsers(dest='system_type')

    lora_lstm_parser = subparser.add_parser("lora_LSTM")


    ############### for content encoder ####################################################

    for system_type_parser in [lora_lstm_parser]:
        subparser_c = system_type_parser.add_subparsers(dest='system_type_c')

        whisper_parser = subparser_c.add_parser("whisper")

    args = parser.parse_args()

    model_dir_1 = 'saved_models'
    system_flag = f'-{args.system_type}' if args.system_type != 'lora_LSTM' else ''
    #args.threshold = -np.infty
    args.threshold = -np.inf
    if args.system_type == 'lora_LSTM':
        base_model = LoraLSTM()
    else:
        raise NotImplementedError('Unsupported System Type')
    speaker_encoder = base_model

    # model
    if args.system_type_c == 'whisper':
        content_encoder = WhisperPPG()
        print('load whisper model')
    else:
        raise NotImplementedError
    system_flag_c = f'-{args.system_type_c}' if args.system_type_c != 'whisper' else ''

    
    src_root = './storage/data_svc-all_singers-10_voices'
    # backtrack_flag = '_backtrack' if args.backtrack else ''
    backtrack_flag = '-backtrack' if args.backtrack else ''
    des_root = './storage/data_svc-adver{}{}{}-lr={}'.format(system_flag, system_flag_c, backtrack_flag, str(args.lr).replace('.', '_'))
    if args.src_flag is not None:
        src_root = f'./storage/data_svc-{args.src_flag}'
    if args.des_flag is not None:
        des_root = f'./storage/data_svc-{args.des_flag}'
    print(system_flag, src_root, des_root)

    with torch.no_grad():
        gender_2_imposter_embs = {'F': None, 'M': None}
        gender_2_imposter_idx_2_flag = {'F': {}, 'M': {}}
        for x in os.listdir(src_root):
            y = os.path.join(src_root, x, f'lora_speaker{system_flag}.npy')
            if not os.path.exists(y):
                continue
            z = np.load(y)
            z = torch.from_numpy(z).unsqueeze(0)
            g = x.split('_')[0]
            if gender_2_imposter_embs[g] is None:
                gender_2_imposter_embs[g] = z
            else:
                gender_2_imposter_embs[g] = torch.cat((gender_2_imposter_embs[g], z), dim=0)
            gender_2_imposter_idx_2_flag[g][gender_2_imposter_embs[g].shape[0]-1] = x

    all_spk_keys = sorted(os.listdir(src_root))
    import yaml
    des_file = 'select-target_speakers-source_speeches-des.yaml'
    with open(des_file, 'r') as f:
        spks_keys = list(yaml.load(f, Loader=yaml.FullLoader).keys())
        print(len(spks_keys))
    
    if args.limit_target_spk:
        spks_keys = [x for x in all_spk_keys if x in spks_keys]
    else:
        spks_keys = all_spk_keys
    print(len(spks_keys), spks_keys)
    
    my_masker = Masker(device='cuda')

    back_track_path = './amazing_grace.m4a'
    backtrack, backtrack_sr = librosa.load(back_track_path, sr=16000)
    print('backtrack:', backtrack_sr, backtrack.shape, backtrack.dtype, backtrack.max(), backtrack.min())

    target_wav_path = './cxk-jntm_[cut_2sec].wav'
    target_wav, target_wav_sr = librosa.load(target_wav_path, sr=16000)
    print('target_wav:', target_wav_sr, target_wav.shape, target_wav.shape[0] / target_wav_sr, target_wav.dtype, target_wav.max(), target_wav.min())

    args.end = len(spks_keys) if args.end == -1 else args.end
    for idx_spk, flag in enumerate(spks_keys[args.start:args.end]):

        paths = []
        wavs = []

        try:
            g_dir = os.path.join(src_root, flag, 'waves')
            des_g_dir = os.path.join(des_root, flag, 'waves')
            des_g_dir_backtrack = os.path.join(des_root, flag, 'waves-with_backtrack')
            os.makedirs(des_g_dir, exist_ok=True)
            os.makedirs(des_g_dir_backtrack, exist_ok=True)
            for name in sorted(os.listdir(g_dir))[:args.voice_num]:
                if 'wav' not in name:
                    continue
                src_path = os.path.join(g_dir, name)
                des_path = os.path.join(des_g_dir, name)
                des_path_backtrack = os.path.join(des_g_dir_backtrack, name)
                paths.append((src_path, des_path, idx_spk, des_path_backtrack))
                wav, fs = torchaudio.load(src_path)
                assert fs == 16_000, 'sampling rate is not 16KHZ'
                wav = wav.squeeze(0).cuda()
                wav.requires_grad = True
                wavs.append(wav)
            
            exist = True
            for idx, wav in enumerate(wavs):
                if not os.path.exists(paths[idx][1]):
                    exist = False
            if exist:
                print('Exist, skip')
                continue
            else:
                print(f'start for {idx_spk} {flag}')
            
            with torch.no_grad():
                ori_embs = []
                ori_emb_mean = speaker_encoder.embedding(wavs[0].unsqueeze(0).unsqueeze(1))
                ori_embs.append(ori_emb_mean.clone()) # avoid influcing
                for wav in wavs[1:]:
                    pp_emb = speaker_encoder.embedding(wav.unsqueeze(0).unsqueeze(1))
                    ori_emb_mean += pp_emb
                    ori_embs.append(pp_emb)
                ori_emb_mean /= len(wavs)
                ori_emb_mean = ori_emb_mean.clone()
                ori_emb_mean = ori_emb_mean.squeeze(0)

                my_g = flag.split('_')[0]
                opp_g = 'F' if my_g == 'M' else 'M'
                imposter_embs = gender_2_imposter_embs[opp_g].cuda()
                imposter_idx_2_flag = gender_2_imposter_idx_2_flag[opp_g]
                imposter_sims = speaker_encoder.scoring_trials(imposter_embs, ori_emb_mean.unsqueeze(0))
                imposter_sims = imposter_sims.squeeze(0).detach().cpu().numpy()
                imposter_idx = np.argmin(imposter_sims).flatten()[0]
                imposter_emb = imposter_embs[imposter_idx]
                print(flag, imposter_idx_2_flag[imposter_idx])


            with torch.no_grad():
                # 
                print('compute TH')
                all_theta_array = []
                all_original_max_psd = []
                for wav in wavs:
                    theta_array, original_max_psd = my_masker._compute_masking_threshold(wav.detach().cpu().numpy())
                    backtrack_scale = backtrack / (backtrack.max() / wav.max().item())
                    theta_array_bt, original_max_psd_bt = my_masker._compute_masking_threshold(backtrack_scale[:len(wav)])
                    original_max_psd = max(original_max_psd, original_max_psd_bt) if args.backtrack else original_max_psd
                    theta_array = np.where(theta_array >= theta_array_bt, theta_array, theta_array_bt) if args.backtrack else theta_array
                    theta_array = torch.from_numpy(theta_array).cuda().transpose(0, 1)
                    all_theta_array.append(theta_array)
                    all_original_max_psd.append(original_max_psd)
                all_original_max_psd = torch.tensor(all_original_max_psd).cuda()
                print('compute TH Done')
                origin_wavs = copy.deepcopy(wavs)

            N = args.max_iter
            epsilon = args.epsilon
            step_size = epsilon / 5
            lower_ = -1
            upper_ = 1
            uppers = []
            lowers = []

            for idx, wav in enumerate(wavs):
                upper = torch.clamp(wav+epsilon, max=upper_)
                lower = torch.clamp(wav-epsilon, min=lower_)
                uppers.append(upper)
                lowers.append(lower)

            
            for idx, wav in enumerate(wavs):

                if os.path.exists(paths[idx][1]):
                    print(f'{paths[idx][1]} Exist, skip')
                    continue

                with torch.no_grad():
                    wav_orig = wav.clone()
                    ppg_orig = content_encoder(wav_orig)
                    if len(wav) < len(target_wav):
                        target_wav_my = target_wav[:len(wav)]
                    elif len(wav) > len(target_wav):
                        target_wav_my = np.pad(target_wav, (0, len(wav)-len(target_wav)), mode='wrap')
                    else:
                        target_wav_my = target_wav
                    target_ppg = content_encoder(torch.from_numpy(target_wav_my).float().cuda())

                loss_avg = torch.zeros(4, device=device, dtype=wavs[0].dtype)
                loss_var = torch.zeros(4, device=device, dtype=wavs[0].dtype)

                opt = torch.optim.Adam([wav], lr=args.lr)

                for cur_iter in range(N):

                    wav.grad = None
                    
                    all_loss = torch.zeros(4, device=device, dtype=wavs[0].dtype)
                    loss1 = speaker_encoder(wav.unsqueeze(0).unsqueeze(1), enroll_embs=ori_embs[idx]).squeeze(0)
                    loss2 = speaker_encoder(wav.unsqueeze(0).unsqueeze(1), enroll_embs=imposter_emb.unsqueeze(0)).squeeze(0)
                    all_loss[0] = loss1
                    all_loss[1] = -1. * loss2

                    # th loss
                    delta = wav - origin_wavs[idx]
                    loss_th = my_masker.batch_forward_2nd_stage(delta.unsqueeze(0), all_theta_array[idx].unsqueeze(0), all_original_max_psd[idx:idx+1])
                    all_loss[2] = loss_th

                    # content loss
                    ppg = content_encoder(wav)
                    loss_ppg = torch.nn.functional.cosine_similarity(ppg, target_ppg).mean()
                    all_loss[3] = -1. * loss_ppg

                    # scale loss
                    for l_idx in range(len(all_loss)):
                        loss_avg[l_idx] = loss_avg[l_idx] + (1.0 / (cur_iter+1)) * (all_loss[l_idx].item() - loss_avg[l_idx])
                        loss_var[l_idx] = loss_var[l_idx] + (1.0 / (cur_iter+1)) * ((all_loss[l_idx].item() - loss_avg[l_idx]) ** 2 - loss_var[l_idx])
                        if cur_iter > 0:
                            if loss_var[l_idx] == 0.:
                                loss_var_eps = 1e-8
                            else:
                                loss_var_eps = loss_var[l_idx]
                            all_loss[l_idx] = (all_loss[l_idx] - loss_avg[l_idx]) / (loss_var_eps ** 0.5)
                        else:
                            all_loss[l_idx] = all_loss[l_idx] - loss_avg[l_idx]
                    
                    loss_tol = torch.sum(all_loss)
                    opt.zero_grad()
                    loss_tol.backward()
                    opt.step()

                    wav.data =  torch.clamp(wav.data, lower_, upper_)
                    print('*', flag, idx, cur_iter, loss1.item(), loss2.item(), loss_th.item(), loss_ppg.item(), loss_tol.item())
                    
                
                wav = (wav.detach().cpu().numpy() * (2 ** (16-1) - 1)).astype(np.int16)
                write(paths[idx][1], 16000, wav)

                backtrack_scale = (backtrack / (backtrack.max() / origin_wavs[idx].max().item()))[:len(wav)]
                backtrack_scale = (backtrack_scale * (2 ** (16-1) - 1)).astype(np.int16)
                wav_backtrack = np.stack([backtrack_scale, wav]).T
                write(paths[idx][3], 16000, wav_backtrack)

        except torch.cuda.OutOfMemoryError:
            print('OOM')
            continue
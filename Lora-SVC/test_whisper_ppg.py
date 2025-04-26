import os
import numpy as np
import argparse
import torch
import torchaudio

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,] # [length, dim=1024]
        print(ppg.shape)
        np.save(ppgPath, ppg, allow_pickle=False)

def pred_ppg_t(whisper: Whisper, audio):
    # audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
    ppg = ppg[:ppgln,] # [length, dim=1024]
    # print(ppg.shape)
    return ppg


def pred_ppg_infer_t(whisper: Whisper, audio):
    audln = audio.shape[0]
    ppg_a = []
    idx_s = 0
    while (idx_s + 25 * 16000 < audln):
        short = audio[idx_s:idx_s + 25 * 16000]
        idx_s = idx_s + 25 * 16000
        ppgln = 25 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.extend(ppg)
    if (idx_s < audln):
        short = audio[idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.extend(ppg)
    # print(len(ppg_a), torch.stack(ppg_a).shape)
    return torch.stack(ppg_a)

def pred_ppg_infer_t_batch(whisper: Whisper, audio):
    audln = audio.shape[1]
    ppg_a = []
    idx_s = 0
    while (idx_s + 25 * 16000 < audln):
        short = audio[:, idx_s:idx_s + 25 * 16000]
        idx_s = idx_s + 25 * 16000
        ppgln = 25 * 16000 // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel)
        ppg = ppg[:, :ppgln, :]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.append(ppg)
    if (idx_s < audln):
        short = audio[:, idx_s:audln]
        ppgln = (audln - idx_s) // 320
        # short = pad_or_trim(short)
        mel = log_mel_spectrogram(short).to(whisper.device)
        ppg = whisper.encoder(mel)
        ppg = ppg[:, :ppgln, :]  # [length, dim=1024]
        # print(ppg.shape)
        ppg_a.append(ppg)
    # print(len(ppg_a), torch.stack(ppg_a).shape)
    return torch.cat(ppg_a)
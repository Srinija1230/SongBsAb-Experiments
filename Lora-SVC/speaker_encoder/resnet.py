

from speaker_encoder.utils import parse_enroll_model_file, parse_mean_file_2, check_input_range

from speaker_encoder._auto_speech.models import resnet as ResNet
from speaker_encoder._auto_speech.data_objects.audio import normalize_volume_torch, wav_to_spectrogram_torch
from speaker_encoder._auto_speech.data_objects.params_data import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

BITS = 16

class resnet(nn.Module):

    def __init__(self, model_name, extractor_file, mean_file=None, feat_mean_file=None, feat_std_file=None, 
                model_file=None, threshold=None, device='cpu',
                ):

        super().__init__()

        self.device = device

        checkpoint = torch.load(extractor_file)
        MODEL_NUM_CLASSES = checkpoint['state_dict']['classifier.weight'].shape[0]
        MODEL_NAME = model_name
        self.encoder = eval('ResNet.{}(num_classes={}, pretrained=False)'.format(MODEL_NAME, MODEL_NUM_CLASSES))
        # load checkpoint
        self.encoder.load_state_dict(checkpoint['state_dict'])
        self.encoder.to(device).eval()
        assert self.encoder.training == False

        self.emb_mean = parse_mean_file_2(mean_file, device)

        # feature mean and std
        if feat_mean_file is not None:
            self.feat_mean = torch.load(feat_mean_file, map_location=self.device)
            assert self.feat_mean.shape[0] == n_fft // 2 + 1
        else:
            self.feat_mean = 0
        if feat_std_file is not None:
            self.feat_std = torch.load(feat_std_file, map_location=self.device)
            assert self.feat_std.shape[0] == n_fft // 2 + 1
        else:
            self.feat_std = 1
        
        if model_file is not None:
            self.num_spks, self.spk_ids, self.z_norm_means, self.z_norm_stds, self.enroll_embs = \
                parse_enroll_model_file(model_file, self.device)

        # If you need SV or OSI, must input threshold
        self.threshold = threshold if threshold else -np.infty # Valid for SV and OSI tasks; CSI: -infty

        self.allowed_flags = sorted([
            0, 1, 2
        ])
        self.range_type = 'scale'

    
    def raw(self, x):
        """
        x: (B, 1, T)
        """
        x = x.squeeze(1)
        x = normalize_volume_torch(x, audio_norm_target_dBFS, increase_only=True) # (batch, time)
        feats = wav_to_spectrogram_torch(x) # (batch, frames, #bin)
        return feats

    
    def cmvn(self, feats):
        feats = (feats - self.feat_mean) / self.feat_std
        return feats
    

    def extract_emb(self, x):
        emb = self.encoder(x) - self.emb_mean
        return emb
    

    def compute_feat(self, x, flag=1):
        """
        x: [B, 1, T]
        flag: the flag indicating to compute what type of features
        return feats: [B, T, F]
        """
        assert flag in [f for f in self.allowed_flags if f != 0]
        x = check_input_range(x, range_type=self.range_type)

        feats = self.raw(x) # (B, T, F)
        if flag == 1: # calulate ori feat
            return feats
        elif flag == 2: # calulate norm feat
            feats = self.comput_feat_from_feat(feats, ori_flag=1, des_flag=2)
            return feats
        else: # will not go to this branch
            pass

    
    def comput_feat_from_feat(self, feats, ori_flag=1, des_flag=2):
        """
        x: [B, T, F]
        """
        assert ori_flag in [f for f in self.allowed_flags if f != 0]
        assert des_flag in [f for f in self.allowed_flags if f != 0]
        assert des_flag > ori_flag

        if ori_flag == 1 and des_flag == 2:
            return self.cmvn(feats)
        else: # will not go to this branch
            pass

    def embedding(self, x, flag=0):

        assert flag in self.allowed_flags
        if flag == 0:
            # x = check_input_range(x, range_type=self.range_type)
            feats = self.compute_feat(x, flag=self.allowed_flags[-1])
        elif flag == 1:
            feats = self.comput_feat_from_feat(x, ori_flag=1, des_flag=self.allowed_flags[-1])
        elif flag == 2:
            feats = x
        else:
            pass
        emb = self.extract_emb(feats)
        # return emb - self.emb_mean # [B, D]
        return emb # already subtract emb mean in self.extract_emb(feats)
    

    def scoring_trials(self, enroll_embs, embs):

        return F.cosine_similarity(embs.unsqueeze(2), enroll_embs.unsqueeze(0).transpose(1, 2))
    

    def forward(self, x, flag=0, return_emb=False, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        embedding = self.embedding(x, flag=flag)
        
        if not hasattr(self, 'enroll_embs'):
            assert enroll_embs is not None
        enroll_embs = enroll_embs if enroll_embs is not None else self.enroll_embs
        scores = self.scoring_trials(enroll_embs=enroll_embs, embs=embedding)
        if not return_emb:
            return scores
        else:
            return scores, embedding

    
    def score(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        logits = self.forward(x, flag=flag, enroll_embs=enroll_embs)
        scores = logits
        return scores
    

    def make_decision(self, x, flag=0, enroll_embs=None):
        """
        x: wav or acoustic features (raw/delta/cmvn)
        flag: indicating the type of x (0: wav; 1: raw feat; 2: delta feat; 3: cmvn feat)
        """
        scores = self.score(x, flag=flag, enroll_embs=enroll_embs)

        decisions = torch.argmax(scores, dim=1)
        max_scores = torch.max(scores, dim=1)[0]
        decisions = torch.where(max_scores > self.threshold, decisions,
                        torch.tensor([-1] * decisions.shape[0], dtype=torch.int64, device=decisions.device)) # -1 means reject

        return decisions, scores
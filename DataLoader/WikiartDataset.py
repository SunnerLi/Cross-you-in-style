import os
import cv2
import json
import math
import random
import torch
import librosa
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

class WikiartDatasetEvaluate(torch.utils.data.Dataset):
    """
    1. Return clip name and segment index
    2. Do not return image
    3. Lighter
    """
    def __init__(self, music_list, transform, args):
        self.music_list = music_list
        self.transform = transform
        self.regression = args.regression
        self.triplet = args.triplet
        self.npy = args.npy

        self.sr = args.sr
        self.n_mels = args.n_mels
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.audio_length = args.audio_length
        self.win_sizes = args.win_sizes
        self.year_mean = args.mean
        self.year_std = args.std
        self.split_num = args.split_num

        self.year_base = int(args.year_base)
        self.year_interval = int(args.year_interval)

        print ("Length:{}".format(len(self.music_list)))

    def get_mel_spectrogram(self, music_path, music_npy_path, offset):
        mel_list = []
        
        for win_size in self.win_sizes: 
            # 1. Read .mp3
            try:
                if self.npy == True:
                    offset = 0
                    y = np.load(music_npy_path)[int(offset*self.sr) : int((offset+self.audio_length)*self.sr)]
                else:
                    y, sr = librosa.core.load(music_path, mono=True, sr=self.sr, offset=offset, duration=self.audio_length)
            except Exception as e:
                print ("Failed to load {}:{}".format(music_path, e))
                print ("Use dummy audio")
                y = np.zeros(self.sr*self.audio_length)

            # 2. Convert to mel spectrogram
            _mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=win_size, hop_length=self.hop_length, n_mels=self.n_mels)
            mel_list.append(_mel[None, :, :])

        #  Merge mel spectrogram
        mel = np.concatenate(mel_list, axis=0)
        return mel

    def __getitem__(self, idx):
        # 0. Metadata
        music_path = self.music_list[idx]["path"]
        offset = self.music_list[idx]["third"]
        seg_idx = self.music_list[idx]["seg_idx"]
        name = self.music_list[idx]["name"]

        # 1. Get the mel spec
        mel = self.get_mel_spectrogram(music_path, None, offset)

        # 2. Split along the time-axis into 3 channels
        if self.split_num:
            mel = np.concatenate(np.split(mel, self.split_num, axis=-1), axis=0)

        # 3. Return
        return {"music":torch.from_numpy(mel), 
                "seg_idx":seg_idx,
                "name":name
                }

    def __len__(self):
        return len(self.music_list)

import os
import json
import copy
import torch
import librosa
import torchvision
import torch.nn as nn
import multiprocessing as mp
import torch.nn.functional as F
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader

from DataLoader.WikiartDataset import WikiartDatasetEvaluate

def getLoaderEvaluate(music_path, transform, args, batch_size):
    shuffle = False

    try:
        with open(music_path, "r") as fp: 
            music_list = json.load(fp)
    except Exception as e:
        print ("Failed to load {}:{}".format(music_path, e))

    dataset=WikiartDatasetEvaluate(music_list, transform, args)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers)
    
    return dataloader

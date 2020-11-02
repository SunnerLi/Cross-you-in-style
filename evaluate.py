import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import argparse
import torch
import os
import numpy as np
import json
import tqdm
import sys

import Utils
from Models.model import SVN
from DataLoader.GetLoader import getLoaderEvaluate
from DataLoader.DatasetUtils import getTransform

"""
Generate music representation.
Input params:
    <base> <count> : We will inference <count> times, starting from <base> to <base+count>
    --load_model_path : Load the pretrained model from this directory
    --save_output_path : Write the music representation to this directory
    --sample_train_music_json : The json file that saves data of inferenced clips.
        [{"third":The start time(sec),
          "name": Name of this clip,
          "seg_idx: The index of this clip",
          "path" : Path to the .wav },
          {},{},...]
Output:
    Write the music representation to <--save_output_path>/<name>/Stype_sample<base+count>/<seg_idx>.jpg
"""


def train(args, count, base):
    # Create the folder to save the result
    print ("\t\t", args.save_output_path)
    Utils.mkdir(args.save_output_path)

    # Loader
    transform = getTransform(args)
    evaluate_loader = getLoaderEvaluate(music_path = args.sample_train_music_json,
                               transform = transform,
                               args = args,
                               batch_size = 1)
    Utils.Log ("Sample dataset set: {}".format(len(evaluate_loader)))

    # Load the pretrained model
    model = SVN(
        z_dim = args.z_dim, 
        image_size = min(args.paint_crop_H, args.paint_crop_W), 
        out_class = 54, 
        regression = args.regression, 
        triplet = args.triplet
    )
    model = model.to('cuda')
    model.load(args.load_model_path)
    model.eval()

    # Inference
    name_list = []
    print ("Round {}".format(count+base))
    for i, meta in tqdm.tqdm(enumerate(evaluate_loader)):
        music = meta["music"]
        seg_idx = meta["seg_idx"]
        name = meta["name"]

        music = music.float().cuda()
        seg_idx = np.squeeze(seg_idx.cpu().data.item())
        name = name[0]

        if len(name_list) == 0:
            name_list.append(name)
        else:
            if name_list[-1] != name:
                name_list.append(name)
                # Resample
                print ("Resample Z")
                model.resample()

        # Forward 
        _, pred_paint, _, _ = model(music)

        pred_paint = np.squeeze(pred_paint.cpu().data.numpy())

        path = os.path.join(args.save_output_path, name, "Style_sample{:02d}".format(count+base))
        os.system("mkdir -p {}".format(path))
        paint_name = "{:06d}.jpg".format(seg_idx)
        Utils.dumpPaint(pred_paint, path, 0, 0, name=paint_name)

    print ("Finished")

if __name__ == '__main__':
    args = Utils.getArgs()
    COUNT = args.eva_count
    BASE = args.eva_base
    for count in range(COUNT):
        train(args, count, base=BASE)

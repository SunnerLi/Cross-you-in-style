import json
import torch
import argparse
import librosa
import os
import cv2
import numpy as np
from torchvision.utils import save_image

def getArgs():
    parser = argparse.ArgumentParser()
 
    # Json path
    parser.add_argument("--train_music_json", type=str, default='./Jsons/portrait/train_music.json', help="Path to train.json.")
    parser.add_argument("--val_music_json", type=str, default='./Jsons/portrait/val_music.json', help="Path to val.json.")
    parser.add_argument("--test_music_json", type=str, default='./Jsons/portraittest_music.json', help="Path to test.json.")
    parser.add_argument("--sample_train_music_json", type=str, default='./Jsons/portrait/sample_train_music.json', help="Path to train.json.")
    parser.add_argument("--sample_val_music_json", type=str, default='./Jsons/portrait/sample_val_music.json', help="Path to val.json.")

    parser.add_argument("--train_paint_json", type=str, default='./Jsons/portrait/train_paint.json', help="Path to train.json.")
    parser.add_argument("--val_paint_json", type=str, default='./Jsons/portrait/val_paint.json', help="Path to val.json.")
    parser.add_argument("--test_paint_json", type=str, default='./Jsons/portrait/test_paint.json', help="Path to test.json.")
    parser.add_argument("--sample_train_paint_json", type=str, default='./Jsons/portriat/sample_train_paint.json', help="Path to train.json.")
    parser.add_argument("--sample_val_paint_json", type=str, default='./Jsons/portrait/sample_val_paint.json', help="Path to val.json.")

    # Image
    parser.add_argument("--paint_resize_min_edge", type=int, default=300)
    parser.add_argument("--paint_crop_H", type=int, default=256)
    parser.add_argument("--paint_crop_W", type=int, default=256)

    # Audio
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=1024)
    # parser.add_argument("--hop_length", type=int, nargs="+", default=[256, 426, 512])
    # parser.add_argument("--audio_length", type=float, nargs="+", default=[3.0, 5.0, 6.0])
    # parser.add_argument("--win_sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--win_sizes", type=int, nargs="+", default=[1024])
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--audio_length", type=float, default=2.97)

    # Year
    parser.add_argument("--year_base", type=int, default=1730)
    parser.add_argument("--year_interval", type=int, default=10)

    # Save model path
    # parser.add_argument("--save_model_path", type=str, default="./Result/", help="Path to save the models.")
    parser.add_argument("--save_output_path", type=str, default="./Result/", help="Path to save the whole outputs.")
    parser.add_argument("--load_model_path", type=str, default="", help="Path to load the models.")
    #parser.add_argument("--load_model_epoch", type=int, default="-1", help="Path to load the models.")

    # blahhhhh
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--sample_epoch", type=int, default=5)
    parser.add_argument("--save_model_epoch", type=int, default=5)
    parser.add_argument("--tfboard_log_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2], help="Which gpu to use.")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--npy", action='store_true')

    # Experiment strategy
    parser.add_argument("--regression", action = 'store_true', help = 'If use regression in ACGAN')
    parser.add_argument("--mean", type = float, default = 1830, help = 'The year mean of normalization')
    parser.add_argument("--std", type = float, default = 60.553, help = 'The year standard deviation of normalization')
    parser.add_argument("--triplet", action = 'store_true', help = 'If use regression in ACGAN')
    parser.add_argument("--split_num", type = int, default = 0, help = 'How many fragment you want to split along time axis. ')

    # Evaluation
    parser.add_argument("--eva_base", type=int, default=1)
    parser.add_argument("--eva_count", type=int, default=1)

    args = parser.parse_args()
    presentParameters(vars(args))
    return args

def presentParameters(args_dict):
    """
        Print the parameters setting line by line
        
        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    Log("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        Log("{:>25} : {}".format(key, args_dict[key]))
    Log("===============================")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gridTranspose(img_list):
    """
        Transpose the tensor in grid view
        Arg:    img_list    (List)  - Contains tensors, and the shape is [(BCHW), (BCHW), ... for k times]
        Ret:    The transposed tensor, and the shape is (B*k, C, H, W)
    """
    # Check if the whole tensor has the same shape
    for i, tensor in enumerate(img_list):
        if i > 0:
            if tensor.size() != img_list[i-1].size():
                raise Exception("Size {} and {}. You should make sure the tensor you want to transpose has the same shape!".format(
                    tensor.size(), img_list[i-1].size()
                ))

    # Transpose
    result = []
    num_tensor = len(img_list)          # The number of different music samples
    num_obj = img_list[0].size(0)       # The number of object in each music sample
    for i in range(num_obj):
        for j in range(num_tensor):
            result.append(img_list[j][i])
    result = torch.stack(result, 0)
    assert len(result.size()) == 4
    return result

def dumpWav(mel, sr, n_fft, hop_length):
    return librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=n_fft, hop_length=hop_length)

def label2year(year_tag, args):
    y_base = args.year_base
    y_int = args.year_interval
    
    return year_tag * y_int + y_base

def dumpPaint(paint, path, epoch, batch, itr=0, name=None):
    std = 0.5
    mean = 0.5
    paint = paint*std+mean
    paint *= 255.0
    paint = np.transpose(paint, (1, 2, 0))
    if name == None:
        filename = os.path.join(path, "paint_{}_{}_{}.jpg".format(epoch, batch, itr))
    else:
        filename = os.path.join(path, name)
    cv2.imwrite(filename, paint.astype(np.uint8))

def dumpMal(music, path, epoch, batch, itr=0):
    filename = os.path.join(path, "mel_{}_{}_{}".format(epoch, batch, itr))
    np.save(filename, music)

def dumpYear(year_label, args, path, epoch, batch, itr=0):
    dump = {"year":label2year(year_label, args).tolist()}
    filename = os.path.join(path, "year_{}_{}_{}.json".format(epoch, batch, itr))
    with open(filename, "w") as fp:
        json.dump(dump, fp, indent=4)

def dumpMusicLatent(m_latent, path, epoch, batch, itr=0, name=None):
    if name == None:
        filename = os.path.join(path, "lt_{}_{}_{}".format(epoch, batch, itr))
    else:
        filename = os.path.join(path, name)
    np.save(filename, m_latent)

def sampleGt(dataloader, args):
    """
    Sample ground truth with dataloader with batch_size=1
    """
    for b, meta in enumerate(dataloader):
        music = meta["music"]
        paint = meta["paint"]
        year  = meta["year"]

        music = np.squeeze(music.cpu().data.numpy())
        paint = np.squeeze(paint.cpu().data.numpy())
        year = np.squeeze(year.cpu().data.numpy())

        path = os.path.join(args.save_output_path, "{:02d}".format(b), "GT")
        mkdir(path)

        # Sample
        Log ("Sample to {}".format(path))
        dumpMal(music, path, b, b)
        dumpPaint(paint, path, b, b)
        dumpYear(year, args, path, b, b)

def sample(model, dataloader, epoch, args):
    """
    Sample output with dataloader with batch_size=1
    """
    with torch.no_grad():
        grid_result_list = []
        gt_list = []
        for b, meta in enumerate(dataloader):
            music = meta["music"]
            paint = meta["paint"]
            year  = meta["year"]

            music = music.float().cuda()

            # Forward 
            music = torch.cat([music[:, :, :, :music.size(3)//3], music[:, :, :, music.size(3)//3:2*music.size(3)//3], music[:, :, :, 2*music.size(3)//3:]], 1)
            pred_music, pred_paint, pred_year, m_latent = model(music)

            pred_music = np.squeeze(pred_music.cpu().data.numpy())
            pred_paint = np.squeeze(pred_paint.cpu().data.numpy())
            pred_year = np.squeeze(pred_year.cpu().data.numpy())
            m_latent = np.squeeze(m_latent.cpu().data.numpy())

            path = os.path.join(args.save_output_path, "{:02d}".format(b))

            # Sample
            Log ("Epoch:[{}]. Sample to {}".format(epoch, path))
            dumpMal(pred_music, path, epoch, b)
            dumpPaint(pred_paint, path, epoch, b)
            dumpYear(pred_year, args, path, epoch, b)
            dumpMusicLatent(m_latent, path, epoch, b)

            # Sample for several times
            if b % 5 == 0:
                gt_list.append(paint[0].cpu())
                _, pred_paint, _, _ = model(torch.cat([music] * 9, 0))
                grid_result_list.append(pred_paint.cpu())

        # Save grid
        # grid_result_list = [torch.stack(gt_list, 0)] + grid_result_list
        grid_result_list = [torch.stack(gt_list, 0)] + [gridTranspose(grid_result_list)]
        grid_result = torch.cat(grid_result_list, 0)
        filename = os.path.join(args.save_output_path, "Grid", "{}.jpg".format(epoch))
        save_image(grid_result, filename, normalize=True, nrow=len(gt_list))

###################################################################################3
#   Define the function to compute style
###################################################################################3
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def get_gram(img, vgg):
    assert torch.min(img) >= 0.0 and torch.max(img) <= 255.0
    img = img.repeat(1, 1, 1, 1).cuda()
    features_style = vgg(normalize_batch(img))
    gram_style = [gram_matrix(y) for y in features_style]
    return gram_style


def Log(string):
    print("[MbST]  {}".format(string))

if __name__ == "__main__":
    sr, n_fft, hop_length = 22050, 1024, 512
    y, sr = librosa.core.load("mezame.wav", sr=sr)

    print ("Mel spec")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    re_y = dumpWav(mel, sr, n_fft, hop_length)

    print ("Dump file")
    librosa.output.write_wav("re_mezame.wav", y=y, sr=sr)

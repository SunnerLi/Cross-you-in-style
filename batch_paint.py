import argparse
import os
import os.path as osp
import glob

def mkdir(path):
    os.system("mkdir -p {}".format(path))

parser = argparse.ArgumentParser()
parser.add_argument("--content_image", type=str, required=True, help="Path of the content image.")
parser.add_argument("--style_images", type=str, required=True, help="Path of the folder where style images stay.")
args = parser.parse_args()

#baselines={"CLS_TRIP":"TestGit/MbstExb/evaluateResultClsTrip/Clips/Sarabande/Linear/Style_*/*.jpg", "STYLE":"TestGit/MbstExb/evaluateResultStyle/Clips/Sarabande/Linear/Style_*/*.jpg", "FULL":"TestGit/MbstExb/evaluateResultRegTrip5600/Clips/Sarabande/Linear/Style_*/*.jpg"}
baselines={"FULL":"MmMbst/DebugOutput/Spring/Style_sample_mini/*.jpg"}
t_list = ["transfer", "filtered", "smooth"]
image_name = args.content_image.split("/")[-1].split(".")[0]

#for model_name, lr_style_path in baselines.items():
#save_path = os.path.join(image_name, model_name)
save_path = image_name
lr_style_path = os.path.join(args.style_images, "*.jpg")
os.system("mkdir -p {}".format(save_path))

print ("Copy content to {}/Content".format(save_path))
os.system("mkdir -p {}/Content".format(save_path))
cmd = "cp {} {}/Content/{:>04d}.jpg".format(args.content_image, save_path, 0)
os.system(cmd)

# Collect LR
os.system("mkdir -p {}/LR".format(save_path))
for i, path in enumerate(sorted(glob.glob(lr_style_path))):
    cmd = "cp {} {}/LR/{:>04d}.jpg".format(path, save_path, i)
    os.system (cmd)

# ESRGan
hr_style_path = os.path.join(save_path, "HR")
os.system("mkdir -p {}".format(hr_style_path))
f_in = os.path.abspath("{}/LR".format(save_path))
f_out = os.path.abspath(hr_style_path)
cmd = "cd ESRGAN; python3 test.py --f_in {} --f_out {}".format(f_in, f_out)
print ("."*20)
print (cmd)
print ("."*20)
os.system (cmd)

# Linear Style Transfer
hr_style_path = os.path.join(save_path, "HR")
f_out = os.path.abspath(hr_style_path)
s_in = f_out + "/"
#c_in = os.path.abspath(os.path.join(save_path, "Content")) + "/"
c_in = os.path.abspath("{}/Content/{:>04d}.jpg".format(save_path, 0))
f_out = os.path.abspath(os.path.join(save_path, "Result")) + "/"
mkdir(f_out)

cmd = "cd LinearStyleTransfer; python3 TestPhotoReal.py --stylePath {} --contentPath {} --outf {}".format(s_in, c_in, f_out)
print ("."*20)
print (cmd)
print ("."*20)
os.system (cmd)

# Split
print ("."*20)
print ("Move images to {}/{}".format(save_path, t_list))
print ("."*20)
for result_name in glob.glob(f_out+"*.png"):
    for _type in t_list:
        if _type in result_name:
            tar = os.path.join(save_path, _type)
            if osp.isdir(tar) == False:
                cmd = "mkdir -p {}".format(tar)
                os.system(cmd)

            cmd = "cp {} {}".format(result_name, tar)
            os.system(cmd)


import os
import glob

# PATH = "/home/lr/kawamoto/m1/GradeSystem/essay/"
PATH = "/home/lr/kawamoto/m1/GradeSystem/essay/wi+locness/"

input_files = glob.glob(PATH+'original/**/*.raw')
input_files.sort()
exist_lst = set(glob.glob(PATH+"nngec/**/*.raw"))

os.chdir("/raid/zhang/GEC/gector")

# gec
for file in input_files:
    out_file = file.replace("original", "nngec")
    if out_file not in exist_lst:
        print(out_file)
        os.system(
            f"CUDA_VISIBLE_DEVICES=1 python /raid/zhang/GEC/gector/predict.py --model_path /raid/zhang/GEC/gector/shared_models/xlnet_0_gector.th --input_file {file} --output_file {out_file} --is_ensemble 1 --batch_size 512 --beam_size 1")

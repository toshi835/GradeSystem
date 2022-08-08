import glob
import json
import os
import re

PATH = "/home/lr/kawamoto/m1/GradeSystem/essay/wi+locness/"
files = glob.glob(
    PATH+'json/*.json')
files.sort()

# jsonファイルからcefrランクを入手します
print("CEFR rank")
cefr_lsts = []
for file in files:
    ce_lst = []
    print(file)
    with open(file, "r") as lines_json:
        # df = json.load(f_json)
        # この場合decodeエラーが出てしまう
        # {"text":~~},のこのコロンがないため。
        # なので1行ずつ処理する
        for line in lines_json:
            df = json.loads(line)
            cefr = re.sub("\...?|\+", "", df["cefr"])
            ce_lst.append(cefr)
    cefr_lsts.append(ce_lst)
    print(len(ce_lst))

# src文を入手
print("src")
files = glob.glob(
    PATH+'text/*.src')
files.sort()
src_lsts = []

for file in files:
    print(file)
    with open(file, "r") as f:
        s = f.read()
        sr_lst = s.strip().strip("@@@").split("@@@")
    src_lsts.append(sr_lst)
    print(len(sr_lst))

print("correct")
# correct文を入手
files = glob.glob(
    PATH+'text/*.correct')
files.sort()

correct_lsts = []
for file in files:
    print(file)
    with open(file, "r") as f:
        s = f.read()
        cor_lst = s.strip().strip("@@@").split("@@@")
    correct_lsts.append(cor_lst)
    print(len(cor_lst))

for rank_lst, src_lst, cor_lst in zip(cefr_lsts, src_lsts, correct_lsts):
    for rank, src, cor in zip(rank_lst, src_lst, cor_lst):
        num = len(os.listdir(PATH+"original/{}/".format(rank)))
        with open(PATH+"original/{}/{:04}.raw".format(rank, num+1), "w") as f_ori:
            f_ori.write(src.strip())
        with open(PATH+"correct/{}/{:04}.raw".format(rank, num+1), "w") as f_cor:
            f_cor.write(cor.strip())

print("A1:", len(os.listdir(PATH+"original/A1/")))
print("A2:", len(os.listdir(PATH+"original/A2/")))
print("B1:", len(os.listdir(PATH+"original/B1/")))
print("B2:", len(os.listdir(PATH+"original/B2/")))
print("C1:", len(os.listdir(PATH+"original/C1/")))
print("C2:", len(os.listdir(PATH+"original/C2/")))

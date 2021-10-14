import glob
import os

# PATH = "/home/lr/kawamoto/m1/GradeSystem/essay/"
PATH = "/home/lr/kawamoto/m1/GradeSystem/essay/wi+locness/"

# NNGEC
# 1. shape gec result
files = glob.glob(PATH+'nngec/**/*.raw')
files.sort()

for file in files:
    print("nngec-1", file)
    with open(file, "r") as f:
        s = f.read()
        s = s.replace("[\"", "")
        s = s.replace("\"]", "")
    with open(file, "w") as f:
        f.write(s)

# 2. merge original and gec result
ori_files = glob.glob(PATH+'original/**/*.raw')
ori_files.sort()

for p_ori in ori_files:
    p_cor = p_ori.replace("original", "nngec")
    p_pair = p_ori.replace("original", "ori_nngec_pairs")
    with open(p_ori, "r") as f_ori, open(p_cor, "r") as f_cor, open(p_pair, "w") as f_pair:
        print("nngec-2", p_ori)
        for ori_line, cor_line in zip(f_ori.readlines(), f_cor.readlines()):
            ori = ori_line.rstrip()
            cor = cor_line.rstrip()
            if ori == "":
                ori = "."
            if cor == "":
                cor = "."

            f_pair.write(ori + "\n")
            f_pair.write(cor + "\n")

# 3. get alignment
input_files = glob.glob(PATH+'ori_nngec_pairs/**/*.raw')
input_files.sort()

for file in input_files:
    print("nngec-3", file)
    out_file = file.replace("ori_nngec_pairs", "ori_nngec_pairs_xml")
    os.system(
        f"ruby2.5 cpsalg.rb <{file} >{out_file}")

# STATGEC
# 2. merge original and gec result
ori_files = glob.glob(PATH+'original/**/*.raw')
ori_files.sort()

for p_ori in ori_files:
    p_cor = p_ori.replace("original", "statgec")
    p_pair = p_ori.replace("original", "ori_statgec_pairs")
    with open(p_ori, "r") as f_ori, open(p_cor, "r") as f_cor, open(p_pair, "w") as f_pair:
        print("stat-2", p_ori)
        for ori_line, cor_line in zip(f_ori.readlines(), f_cor.readlines()):
            ori = ori_line.rstrip()
            cor = cor_line.rstrip()
            if ori == "":
                ori = "."
            if cor == "":
                cor = "."

            f_pair.write(ori + "\n")
            f_pair.write(cor + "\n")

# 3. get alignment
input_files = glob.glob(PATH+'ori_statgec_pairs/**/*.raw')
input_files.sort()

for file in input_files:
    print("stat-3", file)
    out_file = file.replace("ori_statgec_pairs", "ori_statgec_pairs_xml")
    os.system(
        f"ruby2.5 cpsalg.rb <{file} >{out_file}")

# CORRECT
# 2. merge original and gec result
ori_files = glob.glob(PATH+'original/**/*.raw')
ori_files.sort()

for p_ori in ori_files:
    p_cor = p_ori.replace("original", "correct")
    p_pair = p_ori.replace("original", "ori_correct_pairs")
    with open(p_ori, "r") as f_ori, open(p_cor, "r") as f_cor, open(p_pair, "w") as f_pair:
        print("correct-2", p_ori)
        for ori_line, cor_line in zip(f_ori.readlines(), f_cor.readlines()):
            ori = ori_line.rstrip()
            cor = cor_line.rstrip()
            if ori == "":
                ori = "."
            if cor == "":
                cor = "."

            f_pair.write(ori + "\n")
            f_pair.write(cor + "\n")

# 3. get alignment
input_files = glob.glob(PATH+'ori_correct_pairs/**/*.raw')
input_files.sort()

for file in input_files:
    print("correct-3", file)
    out_file = file.replace("ori_correct_pairs", "ori_correct_pairs_xml")
    os.system(
        f"ruby2.5 cpsalg.rb <{file} >{out_file}")

import glob
import os

# 1, shape gec result
files = glob.glob('/home/lr/kawamoto/m1/GradeSystemForEssay/cefrj/nngec/**/*.raw')
files.sort()


for file in files:
    print(1, file)
    with open(file, "r") as f:
        s = f.read()
        s = s.replace("[\"", "")
        s = s.replace("\"]", "")
    with open(file, "w") as f:
        f.write(s)

# 2. merge original and gec result
ori_files = glob.glob('/home/lr/kawamoto/m1/GradeSystemForEssay/cefrj/original/**/*.raw')
ori_files.sort()

for p_ori in ori_files:
    p_cor = p_ori.replace("original", "nngec")
    p_pair = p_ori.replace("original", "ori_nngec_pairs")
    with open(p_ori, "r") as f_ori, open(p_cor, "r") as f_cor, open(p_pair, "w") as f_pair:
        print(2, p_ori)
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
input_files = glob.glob('/home/lr/kawamoto/m1/GradeSystemForEssay/cefrj/ori_nngec_pairs/**/*.raw')
input_files.sort()

for file in input_files:
    print(3, file)
    out_file = file.replace("ori_nngec_pairs", "ori_nngec_pairs_xml")
    os.system(
        f"ruby cpsalg.rb <{file} >{out_file}")

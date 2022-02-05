# textbookを分割する
from email.policy import default
import glob

ranks = ["A1", "A2", "B1", "B2", "C1", "C2"]
all_files = [glob.glob("../textbook/raw/***_{}.raw".format(rank))
             for rank in ranks]

para_static = dict()
para_static["sen_num_sum"] = 0
para_static["sen_num_cnt"] = 0
para_static["sen_num_min"] = float("inf")
para_static["sen_num_max"] = -float("inf")

left_cnt = 0
for rank, files in zip(ranks, all_files):
    output_cnt = 0
    print("original size of {}: {}".format(rank, len(files)))
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            paragraph = text.strip().split("\n\n")
            paragraph = [p.strip() for p in paragraph]
            left = ""
            for p in paragraph:
                lst_p = p.strip().split("\n")
                para_static["sen_num_sum"] += len(lst_p)
                para_static["sen_num_min"] = min(
                    para_static["sen_num_min"], len(lst_p))
                # if para_static["sen_num_max"] < len(lst_p):
                #    para_static["sen_num_max"] = len(lst_p)
                #    print(len(lst_p), file)
                para_static["sen_num_max"] = max(
                    para_static["sen_num_max"], len(lst_p))
                para_static["sen_num_cnt"] += 1

                if left:
                    lst_p = left.split("\n")+lst_p
                    p = left+"\n"+p
                    left = ""

                if len(lst_p) < 5:
                    left = p
                    left_cnt += 1
                    continue

                out_path = "../textbook/paragraph/{}".format(
                    rank) + "/{0:04}.dat".format(output_cnt)
                with open(out_path, "w") as out:
                    out.write(p)
                    output_cnt += 1
        if left:
            out_path = "../textbook/paragraph/raw/{}".format(
                rank) + "/{0:04}.dat".format(output_cnt-1)
            with open(out_path, "a") as out:
                out.write(left)
    print("revised size of {}:  {}".format(rank, output_cnt))

print("---------Static information---------")
print("number of sentense per paragraph")
print("Mean: {:.3f}".format(
    para_static["sen_num_sum"]/para_static["sen_num_cnt"]))
print("Max:  {}".format(para_static["sen_num_max"]))
print("Min:  {}".format(para_static["sen_num_min"]))
print("number of paragraph: {}".format(para_static["sen_num_cnt"]))
print(f"number of paragraph which are less than 5 lines: {left_cnt}")
print("------------------------------------")

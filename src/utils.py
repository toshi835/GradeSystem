import json

import numpy as np
import torch
from torch.utils.data import Dataset


def show_output(grade, stats=[], word_diff=[], grmitem=[]):
    grade_class = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C1"}
    output_dic = dict()
    output_dic["grade"] = grade_class[grade[0]]
    output_dic["stats"] = stats
    output_dic["word_diff"] = word_diff
    output_dic["grmitem"] = grmitem

    return output_dic


def data_loader(mode, FEA_PATH="", EMBED_PATH="", wo_ngram=False):
    if FEA_PATH:
        FEA_PATH += f"{mode}.csv"
    if EMBED_PATH:
        EMBED_PATH += f"{mode}_embed.csv"
    if wo_ngram:
        start = 26591
    else:
        start = 0
    x = []
    y = []
    if FEA_PATH and not EMBED_PATH:
        print("loading from ", FEA_PATH)
        with open(FEA_PATH, "r") as f:
            for line in f:
                lines = list(map(float, line.split(",")))
                x.append(np.array(lines[start:-1]))
                y.append(int(lines[-1]))
    elif not FEA_PATH and EMBED_PATH:
        print("loading from ", EMBED_PATH)
        with open(EMBED_PATH, "r") as f:
            for line in f:
                lines = list(map(float, line.split(",")))
                x.append(np.array(lines[start:-1]))
                y.append(int(lines[-1]))
    else:
        print("loading from ", FEA_PATH, EMBED_PATH)
        with open(FEA_PATH, "r") as f, open(EMBED_PATH, "r") as add_f:
            for line, add_line in zip(f, add_f):
                lines = list(map(float, line.split(",")))
                add_lines = list(map(float, add_line.split(",")))
                x.append(np.array(lines[start:-1] + add_lines[:-1]))
                y.append(int(add_lines[-1]))
                assert int(lines[-1]) == int(add_lines[-1])

    x = np.array(x)
    y = np.array(y)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    return x, y


class CreateDataset(Dataset):
    def __init__(self, file_path, feature_file_path="", gec_file_path=""):
        self.file_path = file_path  # src_ids
        self.feature_file_path = feature_file_path  # feature
        self.gec_file_path = gec_file_path  # gec_feature
        self.x = []
        self.y = []
        self.feature = []

        self._build()
        self.split_num = len(self.y[0])

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        if self.feature_file_path:
            return {
                "ids": self.x[index]["input_ids"],
                "mask": self.x[index]["attention_mask"],
                "feature": self.feature[index],
                "labels": self.y[index],
            }
        else:
            return {
                "ids": self.x[index]["input_ids"],
                "mask": self.x[index]["attention_mask"],
                "labels": self.y[index],
            }

    def _build(self):
        print("Loading from {}".format(self.file_path))
        with open(self.file_path, "r") as f_json:
            for line in f_json:
                df = json.loads(line)
                inp = {
                    "input_ids": torch.tensor(df["src_id"]),
                    "attention_mask": torch.tensor(df["att_mask"]),
                }
                tar = torch.tensor(df["target"])

                self.x.append(inp)
                self.y.append(tar)

        if self.feature_file_path:
            print(
                "Loading from {}".format(
                    self.gec_file_path if self.gec_file_path else self.feature_file_path
                )
            )
            with open(
                self.gec_file_path if self.gec_file_path else self.feature_file_path,
                "r",
            ) as f_csv:
                for line in f_csv:
                    lines = list(map(float, line.split(",")))[:-1]
                    self.feature.append(torch.tensor(lines))

            assert len(self.x) == len(self.feature) == len(self.y)
            print("Loading finished")

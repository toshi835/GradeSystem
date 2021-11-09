import numpy as np
import json
import torch
from torch.utils.data import Dataset


def show_output(grade, stats, word_diff, grmitem):
    grade_class = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1'}
    output_dic = dict()
    output_dic['grade'] = grade_class[grade[0]]
    output_dic['stats'] = stats
    output_dic['word_diff'] = word_diff
    output_dic['grmitem'] = grmitem

    return output_dic


def data_loader(PATH, wo_ngram=False):
    print("loading from ", PATH)
    if wo_ngram:
        start = 26591
    else:
        start = 0
    x = []
    y = []
    with open(PATH, "r") as f:
        for line in f:
            lines = list(map(float, line.split(",")))
            x.append(np.array(lines[start:-1]))
            y.append(int(lines[-1]))

    x = np.array(x)
    y = np.array(y)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    return x, y


class CreateDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.x = []
        self.y = []

        print("Loading from {}".format(self.file_path))
        self._build()
        self.split_num = len(self.y[0])

    def __len__(self):  # len(Dataset)で返す値を指定
        return len(self.y)

    def __getitem__(self, index):  # Dataset[index]で返す値を指定
        return {
            'ids': self.x[index]['input_ids'],
            'mask': self.x[index]['attention_mask'],
            'labels': self.y[index]
        }

    def _build(self):
        with open(self.file_path, "r") as f_json:
            for line in f_json:
                df = json.loads(line)
                inp = {"input_ids": torch.tensor(
                    df["src_id"][:128]), "attention_mask": torch.tensor(df["att_mask"][:128])}
                tar = torch.tensor(df["target"])

                self.x.append(inp)
                self.y.append(tar)

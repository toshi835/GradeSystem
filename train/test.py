import numpy as np
from sklearn.externals import joblib
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils import Surface, GrmItem, Feature, output
from model import MLP

from collections import Counter


def test(args):
    # データ読み込み
    x = []
    y = []
    if args.data == "textbook":
        PATH = "../textbook/10_instance/"
    else:
        PATH = '../cefrj/'

    with open(PATH + "train/test.csv", "r") as f:
        for line in f:
            lines = list(map(float, line.split(",")))
            x.append(np.array(lines[:-1]))
            y.append(int(lines[-1]))

    x = np.array(x)
    y = np.array(y)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    if args.clf == "nn":
        if args.data == "textbook":
            split_num = 4
        else:
            split_num = 3

        x = torch.from_numpy(x).float()
        y = y - 1
        y = torch.from_numpy(y).float()

        model = MLP(args, x.shape[1], split_num)  # split分出力
        # model = MLP(args, x.shape[1], 1)  # 1つにしてしきい値
        model.load_state_dict(torch.load(args.OUT))
        model.eval()

        output = model.forward(x)
        nrow, ncol = output.data.shape

        y_pred = []
        for i in range(nrow):
            y_pred.append(np.argmax(output.data[i, :]))
        y_pred = torch.tensor(y_pred)

        """
        # ------しきい値を利用--------
        output.data = output.data.squeeze()
        if args.data == "textbook":
            threshold = [0.124, 0.317, 0.52, 0.685, 0.748]
            y_pred = torch.zeros(nrow)
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
            y_pred += (threshold[2] < output.data) * 1
            y_pred += (threshold[3] < output.data) * 1
            y_pred += (threshold[4] < output.data) * 1
        else:
            threshold = [0.227, 0.706]
            y_pred = torch.zeros(nrow)
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
        # -----------------------
        """

        print()
        print("accuracy: {0:.4f}".format(accuracy_score(y, y_pred)))
        print("macro f1: {0:.4f}".format(f1_score(y, y_pred, average="macro")))
        print("confusion matrix:")
        print(confusion_matrix(y, y_pred))

    else:
        # モデル読み込み
        clf = joblib.load(args.OUT)
        y_pred = clf.predict(x)

        print()
        print("accuracy: {0:.4f}".format(accuracy_score(y, y_pred)))
        print("macro f1: {0:.4f}".format(f1_score(y, y_pred, average="macro")))
        print("confusion matrix:")
        print(confusion_matrix(y, y_pred))


def test_raw(args):
    # データ読み込み
    data = ''
    with open(args.INPUT, 'r') as f:
        for i in f:
            data += i.rstrip() + ' '

    # 素性作成
    surface = Surface(str(data))
    ngram, stats, diff = surface.features()
    grmitem = GrmItem(str(data))
    grm, pos_ngram, grm_freq = grmitem.features()
    inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()

    # モデル読み込み
    clf = joblib.load(args.OUT)
    grade = clf.predict(inputs)
    print(output(grade, stats, diff, grm_freq))

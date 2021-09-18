import torch
import numpy as np
from collections import Counter
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils import Surface, GrmItem, Feature, output
from model import MLP


def test(args):
    # データ読み込み
    x = []
    y = []
    if args.data == "textbook":
        PATH = "../textbook/10_instance/train/"
    elif args.gec == "nn":
        PATH = '../cefrj/train_nngec/'
    elif args.gec == "stat":
        PATH = '../cefrj/train_statgec/'
    elif args.gec == "correct":
        PATH = '../cefrj/train_correct/'
    else:
        PATH = '../cefrj/train/'

    with open(PATH + "test.csv", "r") as f:
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
            split_num = 5  # 4
        else:
            split_num = 3

        x = torch.from_numpy(x).float()
        y = y - 1
        y = torch.from_numpy(y).float()

        model = MLP(args, x.shape[1], split_num)
        model.load_state_dict(torch.load(args.out))
        model.eval()

        output = model.forward(x)
        nrow, ncol = output.data.shape

        y_pred = []
        for i in range(nrow):
            y_pred.append(np.argmax(output.data[i, :]))
        y_pred = torch.tensor(y_pred)

        print()
        print("accuracy: {0:.4f}".format(accuracy_score(y, y_pred)))
        print("macro f1: {0:.4f}".format(f1_score(y, y_pred, average="macro")))
        print("confusion matrix:")
        print(confusion_matrix(y, y_pred))

    else:
        # モデル読み込み
        clf = joblib.load(args.out)
        y_pred = clf.predict(x)

        print()
        print("accuracy: {0:.4f}".format(accuracy_score(y, y_pred)))
        print("macro f1: {0:.4f}".format(f1_score(y, y_pred, average="macro")))
        print("confusion matrix:")
        print(confusion_matrix(y, y_pred))


def test_raw(args):
    # データ読み込み
    data = ''
    with open(args.input, 'r') as f:
        for i in f:
            data += i.rstrip() + ' '

    # 素性作成
    surface = Surface(str(data))
    ngram, stats, diff = surface.features()
    grmitem = GrmItem(str(data))
    grm, pos_ngram, grm_freq = grmitem.features()
    inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()

    # モデル読み込み
    clf = joblib.load(args.out)
    grade = clf.predict(inputs)
    print(output(grade, stats, diff, grm_freq))

import torch
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from prepro_utils import Surface, GrmItem, Feature
from utils import show_output, data_loader
from model import MLP


def test(args):
    # データ読み込み
    if args.data == "textbook":
        PATH = "../textbook/train/"
    elif args.gec in ["nn", "stat"]:
        PATH = f'../essay/train_{args.gec}gec/'
    elif args.gec == "correct":
        PATH = '../essay/train_correct/'
    else:
        PATH = '../essay/train/'

    x, y = data_loader(PATH+"test.csv", args.wo_ngram)

    # mlp
    if args.clf == "nn":
        split_num = max(y)

        x = torch.from_numpy(x).float()
        y = y - 1
        y = torch.from_numpy(y).float()

        model = MLP(args, x.shape[1], split_num)
        model.load_state_dict(torch.load(args.model))
        model.eval()

        output = model.forward(x)
        nrow, ncol = output.data.shape

        y_pred = []
        for i in range(nrow):
            y_pred.append(np.argmax(output.data[i, :]))
        y_pred = torch.tensor(y_pred)

    # lr
    else:
        clf = joblib.load(args.model)
        y_pred = clf.predict(x)

    print()
    print("accuracy: {0:.4f}".format(accuracy_score(y, y_pred)))
    print("macro f1: {0:.4f}".format(f1_score(y, y_pred, average="macro")))
    print("confusion matrix:")
    print(confusion_matrix(y, y_pred))


def test_raw(args):  # lr w/o GECのみ可能
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
    inputs = Feature(ngram=ngram, pos_ngram=pos_ngram,
                     grmitem=grm, word_difficulty=diff, stats=stats).concat()

    # モデル読み込み
    clf = joblib.load(args.model)
    grade = clf.predict(inputs)
    print(show_output(grade, stats, diff, grm_freq))

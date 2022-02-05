import torch
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data.dataloader import DataLoader

from prepro_utils import Surface, GrmItem, Feature
from utils import show_output, data_loader, CreateDataset
from model import MLP, Linear, DebertaClass


def test(args):
    # データ読み込み
    FEA_PATH = ""
    EMBED_PATH = ""
    if args.data == "textbook":
        FEA_PATH = "../textbook/paragraph/train/"if args.feature else ""
        EMBED_PATH = "../textbook/paragraph/train_bert/" if args.embed else ""
    elif args.data == "wi":
        FEA_PATH = '../essay/wi+locness/train/'if args.feature else ""
        EMBED_PATH = "../essay/wi+locness/train_bert/" if args.embed else ""
    elif args.gec in ["nn", "stat"]:
        FEA_PATH = f'../essay/train_{args.gec}gec/'  # if args.feature else ""
    elif args.gec == "correct":
        FEA_PATH = '../essay/train_correct/'  # if args.feature else ""
    else:
        FEA_PATH = '../essay/train/'if args.feature else ""
        EMBED_PATH = "../essay/train_bert/" if args.embed else ""
    assert FEA_PATH or EMBED_PATH

    x, y = data_loader(mode="test", FEA_PATH=FEA_PATH, EMBED_PATH=EMBED_PATH,
                       wo_ngram=args.wo_ngram)

    # mlp
    if args.clf == "mlp":
        split_num = max(y)

        x = torch.from_numpy(x).float()
        y = y - 1
        y = torch.from_numpy(y).float()

        if EMBED_PATH:
            model = Linear(args, x.shape[1], split_num)
        else:
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


def test_bert(args):
    print("test bert model")
    if args.data == "textbook":
        PATH = "../textbook/paragraph/train_bert/"
    elif args.data == "wi":
        PATH = '../essay/wi+locness/train_bert/'
        # elif args.gec in ["nn", "stat"]:
        #    PATH = f'../essay/train_{args.gec}gec/'
        # elif args.gec == "correct":
        #    PATH = '../essay/train_correct/'
    else:
        PATH = '../essay/train_bert/'

    dataset_test = CreateDataset(PATH+"test.json")  # "train_wi.json")
    test_loader = DataLoader(dataset=dataset_test, batch_size=32)

    split_num = dataset_test.split_num
    model = DebertaClass(split_num)
    model.load_state_dict(torch.load(args.model))
    model.cuda(args.gpus)

    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for data in test_loader:
            ids = data['ids'].cuda(args.gpus)
            mask = data['mask'].cuda(args.gpus)
            labels = data['labels'].cuda(args.gpus)

            outputs = model.forward(ids, mask)

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            golds.extend(labels.tolist())

    print()
    print("accuracy: {0:.4f}".format(accuracy_score(golds, preds)))
    print("macro f1: {0:.4f}".format(f1_score(golds, preds, average="macro")))
    print("confusion matrix:")
    print(confusion_matrix(golds, preds))


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

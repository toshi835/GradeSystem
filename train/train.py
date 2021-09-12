import mord
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch.nn.functional as F

from sklearn.externals import joblib
from model import MLP, evaluate


def train(args):
    x_train = []
    y_train = []
    if args.data == "textbook":
        PATH = "../textbook/10_instance/"
    else:
        PATH = '../cefrj/'
    with open(PATH + "train/train.csv", "r") as f:
        for line in f:
            lines = list(map(float, line.split(",")))
            x_train.append(np.array(lines[:-1]))
            y_train.append(int(lines[-1]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)

    x_dev = []
    y_dev = []
    with open(PATH + "train/dev.csv", "r") as f:
        for line in f:
            lines = list(map(float, line.split(",")))
            x_dev.append(np.array(lines[:-1]))
            y_dev.append(int(lines[-1]))

    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)

    # 学習
    if args.clf == "nn":
        if args.data == "textbook":
            split_num = 4
        else:
            split_num = 3
        x_train = torch.from_numpy(x_train).float()
        y_train = np.identity(split_num)[y_train - 1]  # split分出力
        # y_train = (y_train - 1) / (split_num - 1)  # 1つにしてしきい値
        y_train = torch.from_numpy(y_train).float()

        x_dev = torch.from_numpy(x_dev).float()
        y_dev = np.identity(split_num)[y_dev - 1]  # split分出力
        # y_dev = (y_dev - 1) / (split_num - 1)  # 1つにしてしきい値
        y_dev = torch.from_numpy(y_dev).float()

        model = MLP(args, x_train.shape[1], split_num, criterion=torch.nn.MSELoss())  # split分出力
        # model = MLP(args, x_train.shape[1], 1, criterion=torch.nn.MSELoss())  # 1つにしてしきい値
        print(model)

        # select device
        if torch.cuda.is_available():
            model = model.to(args.visible_gpus)
            x_train = x_train.to(args.visible_gpus)
            y_train = y_train.to(args.visible_gpus)
            x_dev = x_dev.to(args.visible_gpus)
            y_dev = y_dev.to(args.visible_gpus)
            print("device: gpu")
        else:
            print("device: cpu")

        train = TensorDataset(x_train, y_train)
        valid = TensorDataset(x_dev, y_dev)

        train_loader = DataLoader(dataset=train, batch_size=32)
        valid_loader = DataLoader(dataset=valid, batch_size=32)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # 0.02)

        loss_history = []
        for epoch in range(15000):  # 1200):
            train_loss = 0
            model.train()
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                val_loss, val_acc = evaluate(model, valid_loader)
                print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                    epoch + 1, train_loss, val_loss, val_acc))
            loss_history.append(train_loss)

        model.eval()
        output = model.forward(x_dev)
        output.data = output.data.to("cpu")
        nrow, ncol = output.data.shape
        y_dev = y_dev.to("cpu")

        y_pred = []
        y_true = []
        for i in range(nrow):
            y_pred.append(np.argmax(output.data[i, :]))
            y_true.append(np.argmax(y_dev[i, :]))
        y_pred = torch.tensor(y_pred)
        y_dev = torch.tensor(y_true)

        """
        # ------しきい値を決める--------
        ma_f = -1
        if args.data == "textbook":  # 6クラス分類
            threshold = [0] * 5
            # 2つに分割
            y_dev_two = (y_dev >= 3) * 1
            for i in range(300, 700):
                threshold_tmp = i / 1000
                y_pred = torch.zeros(nrow)
                y_pred += (threshold_tmp < output.data) * 1
                f1 = f1_score(y_dev_two, y_pred, average="macro")
                if ma_f < f1:
                    threshold[2] = threshold_tmp
                    ma_f = f1
            former_gold, former_pred = [], []
            latter_gold, latter_pred = [], []
            for d, n in zip(output.data, y_dev):
                if d < threshold[2]:
                    former_pred.append(d)
                    former_gold.append(n)
                else:
                    latter_pred.append(d)
                    latter_gold.append(n - 3)
            former_gold, former_pred = np.array(former_gold), np.array(former_pred)
            latter_gold, latter_pred = np.array(latter_gold), np.array(latter_pred)
            # 3分のしきい値
            nrow = len(latter_gold)
            ma_f = -1
            for i in range(600, 1000):
                for j in range(int(threshold[2] * 1000) + 1, i):
                    threshold_tmp = [j / 1000, i / 1000]
                    y_pred = torch.zeros(nrow)
                    y_pred += (threshold_tmp[0] < latter_pred) * 1
                    y_pred += (threshold_tmp[1] < latter_pred) * 1
                    f1 = f1_score(latter_gold, y_pred, average="macro")
                    if ma_f < f1:
                        threshold[3] = threshold_tmp[0]
                        threshold[4] = threshold_tmp[1]
                        ma_f = f1

            nrow = len(former_gold)
            ma_f = -1
            for i in range(int(threshold[2] * 1000) - 1):
                for j in range(i):
                    threshold_tmp = [j / 1000, i / 1000]
                    y_pred = torch.zeros(nrow)
                    y_pred += (threshold_tmp[0] < former_pred) * 1
                    y_pred += (threshold_tmp[1] < former_pred) * 1
                    f1 = f1_score(former_gold, y_pred, average="macro")
                    if ma_f < f1:
                        threshold[0] = threshold_tmp[0]
                        threshold[1] = threshold_tmp[1]
                        ma_f = f1
            print("threshold", *threshold)
            y_pred = torch.zeros(output.data.shape[0])
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
            y_pred += (threshold[2] < output.data) * 1
            y_pred += (threshold[3] < output.data) * 1
            y_pred += (threshold[4] < output.data) * 1

        else:
            threshold = []
            for i in range(700, 1000):
                for j in range(300):
                    threshold_tmp = [j / 1000, i / 1000]
                    y_pred = torch.zeros(nrow)
                    y_pred += (threshold_tmp[0] < output.data) * 1
                    y_pred += (threshold_tmp[1] < output.data) * 1
                    f1 = f1_score(y_dev, y_pred, average="macro")
                    if ma_f < f1:
                        threshold = [threshold_tmp[0], threshold_tmp[1]]
                        ma_f = f1
            assert threshold
            print("threshold:", threshold[0], threshold[1])  # 0.25, 0.739
            y_pred = torch.zeros(nrow)
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
        # -----------------------
        """

        print()
        print("accuracy: {0:.4f}".format(accuracy_score(y_dev, y_pred)))
        print("macro f1: {0:.4f}".format(f1_score(y_dev, y_pred, average="macro")))
        print("confusion matrix:")
        print(confusion_matrix(y_dev, y_pred))

        torch.save(model.state_dict(), args.OUT)

    else:
        clf = mord.LogisticAT(alpha=0.01)
        clf.fit(x_train, y_train)

        # モデル書き出し
        joblib.dump(clf, open(args.OUT, 'wb'))

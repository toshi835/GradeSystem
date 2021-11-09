import mord
import numpy as np
import torch
import joblib
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model import MLP, DebertaClass, evaluate, calculate_loss_and_accuracy
from utils import data_loader, CreateDataset


def train(args):
    if args.data == "textbook":
        PATH = "../textbook/train/"
    elif args.gec in ["nn", "stat"]:
        PATH = f'../essay/train_{args.gec}gec/'
    elif args.gec == "correct":
        PATH = '../essay/train_correct/'
    else:
        PATH = '../essay/train/'

    x_train, y_train = data_loader(PATH+"train.csv", args.wo_ngram)
    if args.wi:
        # x_train_wi, y_train_wi = data_loader(PATH+"train_wi.csv", args.wo_ngram)
        x_train, y_train = data_loader(
            PATH+"train_wi.csv", args.wo_ngram)  # 今だけ
        #x_train = np.concatenate([x_train_wi, x_train])
        #y_train = np.concatenate([y_train_wi, y_train])
    x_dev, y_dev = data_loader(PATH+"dev.csv", args.wo_ngram)

    # mlpの学習
    if args.clf == "mlp":
        split_num = max(y_train)

        x_train = torch.from_numpy(x_train).float()
        y_train = np.identity(split_num)[y_train - 1]
        y_train = torch.from_numpy(y_train).float()

        x_dev = torch.from_numpy(x_dev).float()
        y_dev = np.identity(split_num)[y_dev - 1]
        y_dev = torch.from_numpy(y_dev).float()

        model = MLP(args, x_train.shape[1],
                    split_num, criterion=torch.nn.MSELoss())
        # model.load_state_dict(torch.load("../models/essay/prepro_wi.pth"))
        print(model)

        # select device
        if torch.cuda.is_available():
            model = model.to(args.gpus)
            x_train = x_train.to(args.gpus)
            y_train = y_train.to(args.gpus)
            x_dev = x_dev.to(args.gpus)
            y_dev = y_dev.to(args.gpus)
            print("device: gpu")
        else:
            print("device: cpu")

        train = TensorDataset(x_train, y_train)
        valid = TensorDataset(x_dev, y_dev)

        train_loader = DataLoader(dataset=train, batch_size=32)
        valid_loader = DataLoader(dataset=valid, batch_size=32)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        loss_history = []
        for epoch in range(args.epoch):
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
        # モデル書き出し
        torch.save(model.state_dict(), args.model)

    # lrの学習
    else:
        clf = mord.LogisticAT(alpha=0.01)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_dev)

        # モデル書き出し
        joblib.dump(clf, open(args.model, 'wb'))

    print()
    print("accuracy: {0:.4f}".format(accuracy_score(y_dev, y_pred)))
    print("macro f1: {0:.4f}".format(
        f1_score(y_dev, y_pred, average="macro")))
    print("confusion matrix:")
    print(confusion_matrix(y_dev, y_pred))


def train_bert(args):
    if args.data == "textbook":
        PATH = "../textbook/train_bert/"
    # elif args.gec in ["nn", "stat"]:
    #    PATH = f'../essay/train_{args.gec}gec/'
    # elif args.gec == "correct":
    #    PATH = '../essay/train_correct/'
    else:
        PATH = '../essay/train_bert/'
    if args.wi:
        assert False  # No implementation

    dataset_train = CreateDataset(PATH+"train.json")
    dataset_dev = CreateDataset(PATH+"dev.json")

    split_num = dataset_train.split_num
    model = DebertaClass(split_num)
    num_parameters = sum(pa.numel() for pa in model.parameters())
    print("number of parameter is", num_parameters)
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss()  # sigmoid+BCEloss(交差エントロピー)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5)

    # select device
    if not torch.cuda.is_available():
        assert False

    model = model.cuda(args.gpus)

    train_loader = DataLoader(dataset=dataset_train, batch_size=32)
    valid_loader = DataLoader(dataset=dataset_dev, batch_size=32)

    # 学習
    log_train = []
    log_valid = []
    for epoch in range(args.epoch):
        model.train()
        for data in train_loader:
            # デバイスの指定
            ids = data['ids'].cuda(args.gpus)
            mask = data['mask'].cuda(args.gpus)
            labels = data['labels'].cuda(args.gpus)

            # 勾配をゼロで初期化
            optimizer.zero_grad()

            # 順伝播 + 誤差逆伝播 + 重み更新
            outputs = model.forward(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 損失と正解率の算出
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, criterion, train_loader, args.gpus)
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, criterion, valid_loader, args.gpus)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # ログを出力
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, loss_train, acc_train, loss_valid, acc_valid))

    # チェックポイントの保存
    torch.save(model.to("cpu").state_dict(),  # cpuに回しておかないと他のgpuに回すときに一度元のGPUに積まれてしまう
               f'{args.model}_checkpoint{epoch + 1}.pt')

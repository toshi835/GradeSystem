import mord
import numpy as np
import torch
import joblib
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from model import MLP, Linear, DebertaClass, evaluate, calculate_loss_and_accuracy
from utils import data_loader, CreateDataset


def train(args):
    FEA_PATH = ""
    EMBED_PATH = ""
    if args.data == "textbook":
        FEA_PATH = "../textbook/paragraph/train/" if args.feature else ""
        EMBED_PATH = "../textbook/paragraph/train_bert/" if args.embed else ""
    elif args.data == "wi":
        FEA_PATH = '../essay/wi+locness/train/' if args.feature else ""
        EMBED_PATH = "../essay/wi+locness/train_bert/" if args.embed else ""
    elif args.gec in ["nn", "stat"]:
        FEA_PATH = f'../essay/train_{args.gec}gec/'  # if args.feature else ""
    elif args.gec == "correct":
        FEA_PATH = '../essay/train_correct/'  # if args.feature else ""
    else:
        FEA_PATH = '../essay/train/' if args.feature else ""
        EMBED_PATH = "../essay/train_bert/" if args.embed else ""
    assert FEA_PATH or EMBED_PATH

    x_train, y_train = data_loader(
        mode="train", FEA_PATH=FEA_PATH, EMBED_PATH=EMBED_PATH, wo_ngram=args.wo_ngram)
    x_dev, y_dev = data_loader(
        mode="dev", FEA_PATH=FEA_PATH, EMBED_PATH=EMBED_PATH, wo_ngram=args.wo_ngram)

    # 使わない
    # if args.wi:
    #    x_train_wi, y_train_wi = data_loader(
    #        FEA_PATH+"train_wi.csv", wo_ngram=args.wo_ngram)
    #    x_train = np.concatenate([x_train_wi, x_train])
    #    y_train = np.concatenate([y_train_wi, y_train])

    # mlpの学習
    if args.clf == "mlp":
        split_num = max(y_train)

        x_train = torch.from_numpy(x_train).float()
        y_train = np.identity(split_num)[y_train - 1]
        y_train = torch.from_numpy(y_train).float()

        x_dev = torch.from_numpy(x_dev).float()
        y_dev = np.identity(split_num)[y_dev - 1]
        y_dev = torch.from_numpy(y_dev).float()

        if EMBED_PATH:  # deberta+素性 or DeBERTaの場合は線形結合層のみ
            model = Linear(
                args, x_train.shape[1], split_num, criterion=torch.nn.MSELoss())
        else:
            model = MLP(
                args, x_train.shape[1], split_num, criterion=torch.nn.MSELoss())

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

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # 0.05

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

            if (epoch + 1) % args.save_epoch == 0:
                torch.save(model.state_dict(),
                           f'{args.model}_epoch{epoch + 1}.pth')

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
        torch.save(model.state_dict(), args.model+".pth")

    # lrの学習
    else:
        clf = mord.LogisticAT(alpha=0.01)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_dev)

        # モデル書き出し
        joblib.dump(clf, open(args.model+".pkl", 'wb'))

    print()
    print("accuracy: {0:.4f}".format(accuracy_score(y_dev, y_pred)))
    print("macro f1: {0:.4f}".format(
        f1_score(y_dev, y_pred, average="macro")))
    print("confusion matrix:")
    print(confusion_matrix(y_dev, y_pred))


def train_bert(args):
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

    dataset_train = CreateDataset(PATH+"train.json")
    dataset_dev = CreateDataset(PATH+"dev.json")

    split_num = dataset_train.split_num
    model = DebertaClass(split_num)
    num_parameters = sum(pa.numel() for pa in model.parameters())
    print("number of parameter is", num_parameters)
    print(model)

    criterion = torch.nn.BCEWithLogitsLoss()  # sigmoid+BCEloss(交差エントロピー)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.lr)

    # select device
    if not torch.cuda.is_available():
        assert False

    model = model.cuda(args.gpus)

    train_loader = DataLoader(dataset=dataset_train, batch_size=8)  # 1
    valid_loader = DataLoader(dataset=dataset_dev, batch_size=8)  # 1

    scaler = GradScaler(enabled=True)  # fp16用

    """
    # deberta+素性を行うためにembeddingを保存するためのコード
    # ---注意---
    # DebertaClassのreturnをencoder_layer.view(-1)にして、batchsizeを1にする必要がある
    def write_embed(loader, output_path):
        with open(output_path, "w") as t:
            for data in tqdm(loader):
                # デバイスの指定
                ids = data['ids'].cuda(args.gpus)
                mask = data['mask'].cuda(args.gpus)
                labels = data['labels'].cuda(args.gpus)

                # 勾配をゼロで初期化
                optimizer.zero_grad()

                # 順伝播 + 誤差逆伝播 + 重み更新
                with autocast(enabled=True):
                    encoder_layer = model.forward(ids, mask)
                    encoder = encoder_layer.cpu().tolist()
                    t.write(",".join(list(map(str, encoder))) +
                            "," + str(int(torch.argmax(labels)+1)) + "\n")

    model.eval()
    with torch.no_grad():
        write_embed(loader=train_loader, output_path=PATH+"train_embed.csv")
        write_embed(loader=valid_loader, output_path=PATH+"dev_embed.csv")
        dataset_test = CreateDataset(PATH+"test.json")
        test_loader = DataLoader(dataset=dataset_test, batch_size=1)
        write_embed(loader=test_loader, output_path=PATH+"test_embed.csv")
    exit()
    """

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
            with autocast(enabled=True):
                outputs = model.forward(ids, mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

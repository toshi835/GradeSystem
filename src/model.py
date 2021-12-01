import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from transformers import DebertaModel


class MLP(torch.nn.Module):
    def __init__(self, args, n_input, n_output, n_hidden=1024, criterion=torch.nn.MSELoss()):
        super(MLP, self).__init__()
        self.criterion = criterion
        self.l1 = torch.nn.Linear(n_input, n_hidden)
        self.l2 = torch.nn.Linear(n_hidden, n_output)
        self.args = args

    def forward(self, x):
        h1 = self.l1(x)
        h2 = torch.relu(h1)
        h3 = self.l2(h2)
        h4 = torch.sigmoid(h3)
        return h4

    def training_step(self, batch):
        x_tr, y_tr = batch
        x_tr = Variable(x_tr)
        y_tr = Variable(y_tr)
        y_pred = self(x_tr).squeeze()
        loss = self.criterion(y_pred, y_tr)
        return loss

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self, batch):
        x_vl, y_vl = batch
        x_vl = Variable(x_vl)
        y_vl = Variable(y_vl)

        output = self(x_vl)
        loss = self.criterion(output.data, y_vl)

        nrow, ncol = output.data.shape
        output = output.data.to("cpu")
        y_vl = y_vl.to("cpu")
        y_pred = []
        y_val = []
        for i in range(nrow):
            y_pred.append(np.argmax(output[i, :]))
            y_val.append(np.argmax(y_vl[i, :]))
        y_pred = torch.tensor(y_pred)
        y_val = torch.tensor(y_val)

        acc = accuracy_score(y_val, y_pred)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        epoch_acc = torch.tensor([x['val_acc']
                                 for x in outputs]).mean()  # Combine accuracies
        return epoch_loss.item(), epoch_acc.item()


class DebertaClass(torch.nn.Module):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/deberta/modeling_deberta.py
    def __init__(self, output_size, drop_rate=0.2):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.dropout = torch.nn.Dropout(drop_rate)

        self.classifier = torch.nn.Linear(768, int(output_size))

    def forward(self, ids, mask):
        outputs = self.deberta(ids, attention_mask=mask)
        encoder_layer = outputs.last_hidden_state[:, 0]
        droped_output = self.dropout(encoder_layer)
        logits = self.classifier(droped_output)
        return logits  # encoder_layer.view(-1)


def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def calculate_loss_and_accuracy(model, criterion, loader, gpus):
    """ 損失・正解率を計算"""
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # デバイスの指定
            ids = data['ids'].cuda(gpus)
            mask = data['mask'].cuda(gpus)
            labels = data['labels'].cuda(gpus)

            # 順伝播
            outputs = model.forward(ids, mask)

            # 損失計算
            loss += criterion(outputs, labels).item()

            # バッチサイズの長さの予測ラベル配列
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total

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


class Linear(MLP):
    def __init__(self, args, n_input, n_output, criterion=torch.nn.MSELoss(), drop_rate=0.1):
        super().__init__(args, n_input, n_output, criterion=criterion)
        self.dropout = torch.nn.Dropout(drop_rate)
        self.classifier = torch.nn.Linear(n_input, n_output)

    def forward(self, x):
        logits = self.classifier(self.dropout(x))
        final_logits = torch.sigmoid(logits)
        return final_logits


class DebertaClass(torch.nn.Module):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/models/deberta/modeling_deberta.py
    def __init__(self, output_size, drop_rate=0.2, feature_length=0):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.dropout = torch.nn.Dropout(drop_rate)

        #self.hidden = torch.nn.Linear(768+feature_length, 1024)  # mlp
        # self.classifier = torch.nn.Linear(1024, int(output_size))  # mlp
        self.classifier = torch.nn.Linear(768+feature_length, int(output_size))
        self.feature_length = feature_length

    def forward(self, ids, mask, feature=None):
        outputs = self.deberta(ids, attention_mask=mask)
        encoder_layer = outputs.last_hidden_state[:, 0]
        droped_output = self.dropout(encoder_layer)
        if self.feature_length:
            logits = self.classifier(torch.cat((droped_output, feature), 1))
        else:
            logits = self.classifier(droped_output)
        return logits
        # return encoder_layer.view(-1)
    """
    # mlp
    def forward(self, ids, mask, feature=None):
        outputs = self.deberta(ids, attention_mask=mask)
        encoder_layer = outputs.last_hidden_state[:, 0]
        droped_output = self.dropout(encoder_layer)

        if self.feature_length:
            h1 = self.hidden(torch.cat((droped_output, feature), 1))
        else:
            h1 = self.hidden(droped_output)
        h2 = torch.relu(h1)
        logits = self.classifier(h2)
        return logits
    """


def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def calculate_loss_and_accuracy(model, criterion, loader, gpus, is_ordinal_regression=False, use_feature=False):
    """ ???????????????????????????"""
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # ?????????????????????
            ids = data['ids'].cuda(gpus)
            mask = data['mask'].cuda(gpus)
            labels = data['labels'].cuda(gpus)
            argmax_labels = torch.argmax(labels, dim=-1).cpu().numpy()
            feature = None
            if use_feature:
                feature = data['feature'].cuda(gpus)

            # ?????????
            outputs = model.forward(ids, mask, feature)

            # ????????????
            if is_ordinal_regression:
                labels = torch.rot90(torch.cumsum(
                    torch.rot90(labels, 2), dim=1), 2)
            loss += criterion(outputs, labels).item()

            # ???????????????????????????????????????????????????
            if is_ordinal_regression:
                rolled = torch.roll(outputs, shifts=-1, dims=1)
                rolled.T[-1] = torch.zeros(len(outputs), device=gpus)
                outputs = outputs-rolled
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            total += len(argmax_labels)
            correct += (pred == argmax_labels).sum().item()

    return loss / len(loader), correct / total

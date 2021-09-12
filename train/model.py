import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


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

        """
        # -------しきい値--------
        output = output.squeeze()
        if self.args.data == "textbook":
            y_val = torch.tensor(y_vl) * 5
            threshold = [0.124, 0.317, 0.52, 0.685, 0.748]
            y_pred = torch.zeros(nrow)
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
            y_pred += (threshold[2] < output.data) * 1
            y_pred += (threshold[3] < output.data) * 1
            y_pred += (threshold[4] < output.data) * 1
        else:
            y_val = torch.tensor(y_vl) * 2
            threshold = [0.25, 0.739]
            y_pred = torch.zeros(nrow)
            y_pred += (threshold[0] < output.data) * 1
            y_pred += (threshold[1] < output.data) * 1
        # ----------------------
        """

        acc = accuracy_score(y_val, y_pred)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        epoch_acc = torch.tensor([x['val_acc'] for x in outputs]).mean()  # Combine accuracies
        return epoch_loss.item(), epoch_acc.item()


def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

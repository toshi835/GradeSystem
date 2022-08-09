import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from transformers import DebertaModel


class MLP(torch.nn.Module):
    def __init__(
        self, args, n_input, n_output, n_hidden=1024, criterion=torch.nn.MSELoss()
    ):
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

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        epoch_acc = torch.tensor(
            [x["val_acc"] for x in outputs]
        ).mean()  # Combine accuracies
        return epoch_loss.item(), epoch_acc.item()


class Linear(MLP):
    def __init__(
        self, args, n_input, n_output, criterion=torch.nn.MSELoss(), drop_rate=0.1
    ):
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
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.dropout = torch.nn.Dropout(drop_rate)

        # self.hidden = torch.nn.Linear(768+feature_length, 1024)  # mlp
        # self.classifier = torch.nn.Linear(1024, int(output_size))  # mlp
        self.classifier = torch.nn.Linear(768 + feature_length, int(output_size))
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


def calculate_loss_and_accuracy(
    model,
    criterion,
    loader,
    gpus,
    is_ordinal_regression=False,
    use_feature=False,
    is_online_ls=False,
):
    """損失・正解率を計算"""
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            # デバイスの指定
            ids = data["ids"].cuda(gpus)
            mask = data["mask"].cuda(gpus)
            labels = data["labels"].cuda(gpus)
            argmax_labels = torch.argmax(labels, dim=-1).cpu().numpy()
            feature = None
            if use_feature:
                feature = data["feature"].cuda(gpus)

            # 順伝播
            outputs = model.forward(ids, mask, feature)

            # 損失計算
            if is_ordinal_regression and not is_online_ls:
                labels = torch.rot90(torch.cumsum(torch.rot90(labels, 2), dim=1), 2)
            loss += criterion(outputs, labels).item()

            # バッチサイズの長さの予測ラベル配列
            if is_ordinal_regression:
                rolled = torch.roll(outputs, shifts=-1, dims=1)
                rolled.T[-1] = torch.zeros(len(outputs), device=gpus)
                outputs = outputs - rolled
            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            total += len(argmax_labels)
            correct += (pred == argmax_labels).sum().item()

    return loss / len(loader), correct / total


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    """

    def __init__(
        self,
        alpha: float,
        n_classes: int,
        smoothing: float = 0.1,
        is_ordinal_regression: bool = False,
    ):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= alpha <= 1, "Alpha must be in range [0, 1]"
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal LS for first epoch
        self.register_buffer("supervise", torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)
        self.is_ordinal_regression = is_ordinal_regression

        # Update matrix is used to supervise next epoch
        self.register_buffer("update", torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer("idx_count", torch.zeros(n_classes))
        if self.is_ordinal_regression:
            self.hard_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: torch.Tensor, y: torch.Tensor):
        # Calculate the final loss
        y_idx = torch.argmax(y, axis=1).to(y.device)
        if y_h.dtype != torch.float32:
            y_h = y_h.to(torch.float32)

        soft_loss = self.soft_loss(y_h, y_idx)

        if self.is_ordinal_regression:
            y = torch.rot90(torch.cumsum(torch.rot90(y, 2), dim=1), 2)
            hard_loss = self.hard_loss(y_h, y)
        else:
            hard_loss = self.hard_loss(y_h, y_idx)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: torch.Tensor, y: torch.Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.
        :param y_h: Predicted logits.
        :param y: Ground truth labels.
        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp().cpu(), y.cpu())
        true_dist = torch.index_select(self.supervise, 1, y.cpu()).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h.cpu(), dim=-1))

    def step(self, y_h: torch.Tensor, y: torch.Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.
        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(
            0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32)
        )

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.
        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()

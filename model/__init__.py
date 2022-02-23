import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F

from PIL import Image
import tqdm.auto as tqdm
import torch
import numpy as np

import pytorch_lightning as pl
import torchmetrics


class TokenOfSeqClassifier(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            nclass=29,
            pad_index=0,
            lr=1e-3,
            weight_decay=1e-2,
            amsgrad=False,  # optim params
            label_idx_to_ignore=[]  # metrics params
    ):
        super(type(self), self).__init__()
        self.model = model
        self.pad_index = pad_index

        # TODO: accurancy removing [EPAD],[BOS],[EOS] tags
        #self.val_pure_acc  = torchmetrics.Accuracy(ignore_index=-100,average='weighted')
        self.val_acc = PureAccuracy(
            label_idx_to_ignore=[*label_idx_to_ignore, self.pad_index])
        self.val_raw_acc = torchmetrics.Accuracy(
            num_classes=nclass, average='weighted', mdmc_average='global', ignore_index=self.pad_index)

        #self.test_acc = torchmetrics.Accuracy(ignore_index=-100,average='weighted')
        #self.test_f1  = torchmetrics.F1Score(ignore_index=-100)

        self.save_hyperparameters('lr', 'weight_decay', 'amsgrad')

        self.val_metrics = {'raw_accuracy': [], 'accuracy': [], 'loss': []}

        self.train_metrics = {'loss': []}

    def forward(self, x):
        return self.module(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(),
                           lr=self.hparams.lr,
                           weight_decay=self.hparams.weight_decay,
                           amsgrad=self.hparams.amsgrad)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = F.nll_loss(pred.transpose(1, 2), y, ignore_index=self.pad_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x).transpose(1, 2)

        loss = F.nll_loss(logits, y, ignore_index=self.pad_index)
        self.log('val_loss', loss, on_epoch=True)

        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_epoch=True,
                 prog_bar=True, logger=True)

        self.val_raw_acc(logits, y)
        self.log('val_raw_acc', self.val_raw_acc, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        self.val_metrics['accuracy'].append(self.val_acc.compute().item())
        self.val_metrics['raw_accuracy'].append(
            self.val_raw_acc.compute().item())
        self.val_metrics['loss'].append(outputs[-1].item())


class PureAccuracy(torchmetrics.Metric):
    def __init__(self, label_idx_to_ignore, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.label_idx_to_ignore = label_idx_to_ignore

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.argmax(dim=1)
        assert preds.shape == target.shape
        m = (target != self.label_idx_to_ignore[0])
        for i in self.label_idx_to_ignore[1:]:
            m &= (target != i)
        self.correct += torch.sum((preds == target)[m])
        self.total += target[m].numel()

    def compute(self):
        return self.correct.float() / self.total

def train_model(
        model, dl_train, dl_test,
        cuda=False, lr=0.001,
        epochs=10, show_plots=False,
        loss_factory=nn.NLLLoss, loss_parms={},
        optimizer_factory=optim.SGD, optimizer_parms={}):

    if show_plots:
        plt.ion()
        plt.show(block=False)

    loss_function = loss_factory(**loss_parms)
    optimizer = optimizer_factory(model.parameters(), lr=lr, **optimizer_parms)

    if cuda:
        model = model.cuda()

    losses = []
    accuracies = []

    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        model.train()
        for x, y in iter(dl_train):
            if cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()

            tag_scores = model(x)
            loss = loss_function(tag_scores.transpose(1, 2), y)

            loss.backward()
            optimizer.step()

        model.eval()
        los, acc = eval_model(model, dl_test, cuda, loss_function)

        losses.append(los)
        accuracies.append(acc)
        # show epoch results
        pbar.set_description(f"Loss:{los:.4f}\tAccurancy:{acc:.4f}")
        if show_plots:
            plt.clf()
            plt.subplot(121)
            plt.title("Test loss")
            plt.plot(losses)

            plt.subplot(122)
            plt.title("Test accuracy")
            plt.plot(accuracies)

            plt.draw()
            plt.pause(0.001)

    if show_plots:
        plt.ioff()

    return losses, accuracies


def eval_model(model, dl_test, cuda, loss_function, return_y=False, return_scores=False):
    model.eval()
    acc = []
    los = []
    if return_y:
        pred = []
        y_true = []
    if return_scores:
        scores = []
    # evaluation
    with torch.no_grad():
        for x, y in iter(dl_test):
            if cuda:
                x, y = x.cuda(), y.cuda()

            tag_scores = model(x)

            loss = loss_function(tag_scores.transpose(1, 2), y)
            los.append(loss.cpu().item())

            acc.append(
                (tag_scores.argmax(dim=2) == y)[y != -100].float()
            )
            if return_y:
                pred.append(
                    (tag_scores.argmax(dim=2))[y != -100].float()
                )

                y_true.append(y[y != -100])
            if return_scores:
                scores.append(tag_scores[y != 100, :])
    acc = torch.cat(acc).mean().item()
    los = np.array(los).mean()
    out = los, acc
    if return_y:
        out = (*out, torch.cat(y_true).cpu().numpy(),
               torch.cat(pred).cpu().numpy())
    if return_scores:
        out = (*out, torch.cat(scores).cpu().numpy())
    return out

def pure_accuracy(model,dl):
    _,_, y_true, y_pred = eval_model(model,dl,torch.cuda.is_available(),nn.NLLLoss(), return_y=True)
    m =  (y_true != _TAGS['[EPAD]'])&(y_true != _TAGS['[BOS]'])&(y_true != _TAGS['[EOS]'])&(y_true != _TAGS['[PAD]'])
    return (y_pred[m]==y_true[m]).mean()
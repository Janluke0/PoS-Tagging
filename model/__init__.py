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
    def __init__(self,
                 module: nn.Module,
                 lr=1e-3, weight_decay=1e-2, amsgrad=False  # optim params
                 ):
        self.module = module
        #TODO: accurancy removing [EPAD],[BOS],[EOS] tags
        #self.val_pure_acc  = torchmetrics.Accuracy(ignore_index=-100,average='weighted')
        self.val_acc = torchmetrics.Accuracy(
            ignore_index=-100, average='weighted')
        self.val_f1 = torchmetrics.F1Score(ignore_index=-100)

        #self.test_acc = torchmetrics.Accuracy(ignore_index=-100,average='weighted')
        #self.test_f1  = torchmetrics.F1Score(ignore_index=-100)

        self.save_hyperparameters('lr', 'weight_decay', 'amsgrad')

    def forward(self, x):
        return self.module(x)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(),
                           lr=self.hparams.lr,
                           weight_decay=self.hparms.weight_decay,
                           amsgrad=self.hparams.amsgrad)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = F.nll_loss(pred.transpose(1, 2), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.model(x).transpose(1, 2)

        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss, on_epoch=True)

        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_epoch=True)

        self.val_f1(logits, y)
        self.log('val_f1', self.val_f1, on_epoch=True)

        return loss

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

import torch
from torch import nn, optim
from torch.nn import functional as F

import pytorch_lightning as pl
import torchmetrics

from positional_encodings import PositionalEncoding1D, Summer

from .. import PureAccuracy



class TokenizedSeq2Seq(nn.Module):

    def __init__(self,
                 d_input,
                 d_model,
                 d_output,
                 nheads=1,
                 dropout=0.05,
                 n_encoder=1,
                 n_decoder=1,
                 d_ff=None,
                 src_embedding=None,
                 tgt_embedding=None,
                 src_pad_idx=None,
                 tgt_pad_idx=None):
        super(type(self), self).__init__()

        self.src_embedding = src_embedding or nn.Sequential(
            nn.Embedding(d_input, d_model),
            Summer(PositionalEncoding1D(d_model)))
        self.tgt_embedding = tgt_embedding or nn.Sequential(
            nn.Embedding(d_output, d_model),
            Summer(PositionalEncoding1D(d_model)))
        d_ff = d_ff or d_model * 4

        self.clf = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(d_model, d_output))
        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nheads,
                                          num_encoder_layers=n_encoder,
                                          num_decoder_layers=n_decoder,
                                          dim_feedforward=d_ff,
                                          dropout=dropout,
                                          batch_first=True)

        self.src_pad = src_pad_idx
        self.tgt_pad = tgt_pad_idx

        self.d_output = d_output

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        #padding masks
        src_pad_mask = (src_pad_mask or (src == self.src_pad)).to(src.device)
        tgt_pad_mask = (tgt_pad_mask or (tgt == self.tgt_pad)).to(tgt.device)

        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        #target mask
        tgt_mask = mk_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer(src_emb,
                               tgt_emb,
                               src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask,
                               tgt_mask=tgt_mask)

        scores = self.clf(out)
        return F.log_softmax(scores, dim=-1)


def mk_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


class PLWrapper(pl.LightningModule):

    def __init__(
            self,
            model: nn.Module,
            lr=1e-3,
            weight_decay=1e-2,
            amsgrad=False,  # optim params
            label_idx_to_ignore=[]  # metrics params
    ):
        super(type(self), self).__init__()
        self.model = model

        # TODO: accurancy removing [EPAD],[BOS],[EOS] tags
        #self.val_pure_acc  = torchmetrics.Accuracy(ignore_index=-100,average='weighted')
        self.val_acc = PureAccuracy(
            label_idx_to_ignore=[*label_idx_to_ignore, self.model.tgt_pad])
        self.val_raw_acc = torchmetrics.Accuracy(
            num_classes=model.d_output,
            average='weighted',
            mdmc_average='global',
            ignore_index=self.model.tgt_pad)

        self.save_hyperparameters('lr', 'weight_decay', 'amsgrad')

        self.val_metrics = {'raw_accuracy': [], 'accuracy': [], 'loss': []}

        self.train_metrics = {'loss': []}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(),
                           lr=self.hparams.lr,
                           weight_decay=self.hparams.weight_decay,
                           amsgrad=self.hparams.amsgrad)

    def training_step(self, train_batch, batch_idx):
        src, tgt = train_batch

        logits = self.model(src, tgt)

        n_logits = torch.zeros(logits.size(0),
                               logits.size(1) + 1,
                               logits.size(2)).to(logits.device)
        n_logits[:, 1:, :] = logits
        n_logits = n_logits[:, :-1, :]
        logits = n_logits

        loss = F.nll_loss(logits.transpose(1, 2),
                          tgt,
                          ignore_index=self.model.tgt_pad)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        src, tgt = val_batch

        logits = self.model(src, tgt)

        # (N,S+1)
        n_logits = torch.zeros(logits.size(0),
                               logits.size(1) + 1,
                               logits.size(2)).to(logits.device)

        # the first element of the column will be zero
        n_logits[:, 1:, :] = logits
        # last element of the sequence is dropped s.t. y.shape==pred.shape
        n_logits = n_logits[:, :-1, :]
        logits = n_logits.transpose(1, 2)

        loss = F.nll_loss(logits, tgt, ignore_index=self.model.tgt_pad)
        self.log('val_loss', loss, on_epoch=True)

        self.val_acc(logits, tgt)
        self.log('val_acc',
                 self.val_acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.val_raw_acc(logits, tgt)
        self.log('val_raw_acc',
                 self.val_raw_acc,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        super().training_epoch_end(outputs)
        self.val_metrics['accuracy'].append(self.val_acc.compute().item())
        self.val_metrics['raw_accuracy'].append(
            self.val_raw_acc.compute().item())
        self.val_metrics['loss'].append(outputs[-1].item())
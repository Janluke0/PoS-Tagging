import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os
from torch import nn
import torch
import torch.nn.functional as F
from positional_encodings import PositionalEncoding1D, Summer


class TokenOfSeqClassifier(nn.Module):
    def __init__(self, d_input, d_model, nclass,
                 nheads=1,  dropout=0.05, N=1,
                 d_ff=None, embedding=None):
        super(type(self), self).__init__()

        self.embedding = embedding or nn.Sequential(
            nn.Embedding(d_input, d_model),
            Summer(PositionalEncoding1D(d_model))
        )
        d_ff = d_ff or d_model*4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nheads, d_ff, dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.att_encoder = nn.TransformerEncoder(
            encoder_layer, N, encoder_norm)
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, nclass)
        )

    def forward(self, x):
        emb = self.embedding(x)
        mask = x == 0
        att = self.att_encoder(emb, src_key_padding_mask=mask)
        scores = self.clf(att)
        return F.log_softmax(scores, dim=-1)

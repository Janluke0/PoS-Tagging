import torch
from torch import nn
import torch.nn.functional as F
from positional_encodings import (PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D,
                                  Summer)

from math import sqrt


class SeqTokenClassifier(nn.Module):
  def __init__(self, d_input, d_model, nclass,
               nheads=1,  dropout=0.05, N=1):
    super(type(self), self).__init__()
    self.embedding = nn.Sequential(
        nn.Linear(d_input, d_model*2),
        nn.Dropout(dropout),
        nn.Linear(d_model*2, d_model),
        Summer(PositionalEncoding1D(d_model))
    )
    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nheads, d_model*4, dropout, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model)
    self.att_encoder = nn.TransformerEncoder(encoder_layer, N, encoder_norm)
    self.clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_model, nclass)
    )

  def forward(self, x):
    emb = self.embedding(
        F.one_hot(x, num_classes=self.embedding[0].in_features).float())
    att = self.att_encoder(emb)
    scores = self.clf(att)  # [:,0])
    return F.log_softmax(scores, dim=-1)


class LSTMSeqTokenClassifier(nn.Module):
  def __init__(self, d_input, d_model, nclass,
               nheads=1,  dropout=0.05, N=1):
    super(type(self), self).__init__()
    self.embedding = nn.LSTM(d_input, d_model, batch_first=True)
    self.pos_enc = Summer(PositionalEncoding1D(d_model))

    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nheads, d_model*4, dropout, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model)
    self.att_encoder = nn.TransformerEncoder(encoder_layer, N, encoder_norm)
    self.clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_model, nclass)
    )

  def forward(self, x):
    emb, _ = self.embedding(
        F.one_hot(x, num_classes=self.embedding.input_size).float())
    emb = self.pos_enc(emb)
    att = self.att_encoder(emb)
    scores = self.clf(att)  # [:,0])
    return F.log_softmax(scores, dim=-1)


class GRUSeqTokenClassifier(nn.Module):
  def __init__(self, d_input, d_model, nclass, clf_gru=False,
               nheads=1,  dropout=0.05, N=1):
    super(type(self), self).__init__()
    self.embedding = nn.GRU(
        d_input, d_model//2, batch_first=True, bidirectional=True)
    self.pos_enc = Summer(PositionalEncoding1D(d_model))

    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nheads, d_model*4, dropout, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model)
    self.att_encoder = nn.TransformerEncoder(encoder_layer, N, encoder_norm)
    self.clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_model, nclass)
    )

  def forward(self, x):
    emb, _ = self.embedding(
        F.one_hot(x, num_classes=self.embedding.input_size).float())
    #print(emb.shape,x.shape)
    emb = self.pos_enc(emb)
    att = self.att_encoder(emb)
    scores = self.clf(att)  # [:,0])
    return F.log_softmax(scores, dim=-1)


class IOGRUSeqTokenClassifier(nn.Module):
  def __init__(self, d_input, d_model, nclass,
               nheads=1,  dropout=0.05, N=1):
    super(type(self), self).__init__()
    self.embedding = nn.GRU(
        d_input, d_model//2, batch_first=True, bidirectional=True)
    self.pos_enc = Summer(PositionalEncoding1D(d_model))

    encoder_layer = nn.TransformerEncoderLayer(
        d_model, nheads, d_model*4, dropout, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model)
    self.att_encoder = nn.TransformerEncoder(encoder_layer, N, encoder_norm)
    self.out_gru = nn.GRU(
        d_model, d_model, batch_first=True, bidirectional=True)
    self.clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_model*2, nclass)
    )

  def forward(self, x):
    emb, _ = self.embedding(
        F.one_hot(x, num_classes=self.embedding.input_size).float())
    #print(emb.shape,x.shape)
    emb = self.pos_enc(emb)
    att = self.att_encoder(emb)
    scores, _ = self.out_gru(att)
    scores = self.clf(scores)
    return F.log_softmax(scores, dim=-1)

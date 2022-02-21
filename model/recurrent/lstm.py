from torch import nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding1D, Summer


class TokenOfSeqClassifier(nn.Module):
  def __init__(self, d_input, d_model, d_output, d_hidden=None, dropout=0.05):
    super(type(self), self).__init__()

    d_hidden = d_hidden or d_model*2

    self.embedding = nn.LSTM(
        d_input, d_model//2, batch_first=True, bidirectional=True)

    self.out_gru = nn.LSTM(
        d_model, d_hidden//2, batch_first=True, bidirectional=True)

    self.clf = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_output)
    )

  def forward(self, x):
    emb, _ = self.embedding(
        F.one_hot(x, num_classes=self.embedding.input_size).float())
    scores, _ = self.out_gru(emb)
    scores = self.clf(scores)
    return F.log_softmax(scores, dim=-1)

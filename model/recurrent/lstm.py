from torch import nn
import torch.nn.functional as F
from model.recurrent import OneHot


class TokenOfSeqClassifier(nn.Module):

    def __init__(self,
                 d_input,
                 d_model,
                 d_output,
                 d_hidden=None,
                 dropout=0.05,
                 add_embedding_projection=False):
        super(type(self), self).__init__()

        d_hidden = d_hidden or d_model * 2

        self.embedding = nn.Embedding(
            d_input, d_model) if add_embedding_projection else OneHot(d_input)

        self.in_lstm = nn.LSTM(
            d_model if add_embedding_projection else d_input,
            d_model // 2,
            batch_first=True,
            bidirectional=True)

        self.out_lstm = nn.LSTM(d_model,
                                d_hidden // 2,
                                batch_first=True,
                                bidirectional=True)

        self.clf = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(d_hidden, d_output))

    def forward(self, x):
        emb = self.embedding(x)
        emb, _ = self.in_lstm(emb)
        scores, _ = self.out_lstm(emb)
        scores = self.clf(scores)
        return F.log_softmax(scores, dim=-1)

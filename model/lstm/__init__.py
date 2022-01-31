from cmath import sqrt
from torch import nn
import torch.nn.functional as F

class LSTMTagger(nn.Module):
  def __init__(self,
               n_tokens, n_tags,
               hidden_dim=32, dropout=0,
               lstm_layers=2,
               bidirectional=False,
               output_layers=1):

    super(LSTMTagger, self).__init__()
    self.n_tokens = n_tokens
    self.n_tags = n_tags
    self.hidden_dim = hidden_dim
    self.bidirectional = bidirectional
    self.dropout = dropout
    self.lstm_layers = lstm_layers
    self.output_layers = output_layers

    assert 0 < self.output_layers < 4

    self.lstm = nn.LSTM(
        input_size=self.n_tokens,
        hidden_size=self.hidden_dim,
        num_layers=self.lstm_layers,
        batch_first=True,
        dropout = self.dropout if self.lstm_layers >1 else 0,
        bidirectional=self.bidirectional
    )

    if bidirectional:
        hidden_dim *=2

    if self.output_layers == 1:
      self.linear_scores =  self.mk_linear_1(hidden_dim, self.n_tags, dropout)
    elif self.output_layers == 2:
      self.linear_scores =  self.mk_linear_2(hidden_dim, self.n_tags, dropout)
    elif self.output_layers == 3:
      self.linear_scores =  self.mk_linear_3(hidden_dim, self.n_tags, dropout)

  def forward(self, sentence_tokens):
    sentence_tokens = F.one_hot(sentence_tokens.long(),self.n_tokens).float()
    lstm_o, _ = self.lstm(sentence_tokens)
    scores = self.linear_scores(lstm_o).float()
    out = F.log_softmax(scores,dim=1)
    return out

  def mk_linear_1(self, input_size, out_size, dropout=0):
    self.ln_o = nn.Linear(input_size, out_size)

    def functional(in_):
      return self.ln_o(in_)
    return functional

  def mk_linear_2(self, input_size, out_size, dropout=0):
    self.ln_in = nn.Linear(input_size, input_size)
    self.ln_o = nn.Linear(input_size, out_size)
    self.dropout = nn.Dropout(dropout)

    def functional(in_):
      a = self.ln_in(in_)
      a = F.relu(a)
      a = self.dropout(a)
      return self.ln_o(a)
    return functional

  def mk_linear_3(self, input_size, out_size, dropout=0):
    h_size = round(abs(sqrt(input_size*out_size)))
    self.ln_in = nn.Linear(input_size, h_size)
    self.ln_h = nn.Linear(h_size, h_size)
    self.ln_o = nn.Linear(h_size, out_size)
    self.dropout = nn.Dropout(dropout)

    def functional(in_):
      a = self.ln_in(in_)
      a = F.relu(a)
      a = self.dropout(a)
      a = self.ln_h(a)
      a = F.relu(a)
      a = self.dropout(a)
      return self.ln_o(a)
    return functional
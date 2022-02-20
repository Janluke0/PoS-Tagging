from torch import nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding1D, Summer


class TokenizedSeq2Seq(nn.Module):
    def __init__(self, d_input, d_model, d_output,
                 nheads=1,  dropout=0.05, N=1,
                 d_ff=None, embedding=None,
                 in_seq_pad=None):
        """
            d_input:
                The # of input tokens
            d_model:
                The internal dimension of the model
            d_output:
                The # of output tokens
            nheads: default = 1
                The # of head in the selfattention layer
            dropout: default = 0.05
            N: default= 1
                The # of stacked layers
            d_ff: default=d_model*4
                The hidden layer size of the selfattention
            embedding: default= [nn.Embedding,PositionalEncoding]
                The embedding layer
            in_seq_pad: int
                The index of the token that will be masked out from the input
        """
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
            nn.Linear(d_model, d_output)
        )

        self.in_seq_pad = in_seq_pad

    def forward(self, x):
        emb = self.embedding(x)
        mask = x == self.in_seq_pad
        att = self.att_encoder(emb, src_key_padding_mask=mask)
        scores = self.clf(att)
        return F.log_softmax(scores, dim=-1)

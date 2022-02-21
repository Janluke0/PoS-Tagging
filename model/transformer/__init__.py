import torch
from torch import nn
import torch.nn.functional as F
from positional_encodings import PositionalEncoding1D, Summer


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

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):

        src_pad_mask = src_pad_mask or (src == self.src_pad)
        tgt_pad_mask = tgt_pad_mask or (tgt == self.tgt_pad)

        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        #This should be done inside the transformer
        #tgt_in  = tgt_emb[:,:-1,:] #drop last token of the sequence
        #tgt_out = tgt_emb[:,1:,:] #drop first  //   //  //   //

        #TODO: apply mask to target

        out = self.transformer(src_emb,
                               tgt_emb,
                               src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask,
                               tgt_mask=mk_mask(tgt.size(1)))

        scores = self.clf(out)
        return F.log_softmax(scores, dim=-1)


def mk_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask
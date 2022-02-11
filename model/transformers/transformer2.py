import torch
from torch import nn
import torch.nn.functional as F
from .attention import PositionalEncoding1d

from .attention import Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, key_dim, value_dim, num_heads):
        super(type(self),self).__init__()

        self.d_k = key_dim
        self.d_v = value_dim
        self.n_h = num_heads
        self.attention = Attention(key_dim, key_dim, value_dim, value_dim, num_heads)
        self.linear = nn.Linear(self.d_v*self.n_h,self.d_v)
        print(self.attention.linear_k)

    def forward(self,key,value,mask=None):
        #key == query
        print('expected', ('batch',self.d_v,'len'))
        print(key.shape, value.shape)
        att = self.attention(key,value,mask).transpose(1,2)
        return att #self.linear(concat)

def addNorm(a, b,**kwargs):
    return F.layer_norm(a+b,a.shape,**kwargs)

class FFN(nn.Module):
    def __init__(self, dim):
        super(type(self), self).__init__()
        self.linear_in = nn.Linear(dim,dim)
        self.linear_out = nn.Linear(dim,dim)

    def forward(self,x):
        hidden = F.relu(self.linear_in(x))
        return self.linear_out(hidden)

Embedding = lambda vocab_size, embedding_dim, scale=1000: nn.Sequential(
        nn.Embedding(vocab_size, embedding_dim),
        PositionalEncoding1d(scale)
)

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(type(self),self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, model_dim, num_heads)
        self.ffn            = FFN(model_dim)

    def forward(self, x):
        ffn_in = addNorm(x, self.self_attention(x,x,x,None))
        return addNorm(ffn_in, self.ffn(ffn_in))

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, N=1):
        super(type(self),self).__init__()
        self.embedding = Embedding(vocab_size,model_dim)
        self.layers    = nn.Sequential(*(EncoderLayer(model_dim, num_heads) for _ in range(N)))

    def forward(self, x):
        return self.layers(self.embedding(x))

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(type(self),self).__init__()
        self.self_attention   = MultiHeadAttention(model_dim, model_dim, num_heads)
        self.encdec_attention = MultiHeadAttention(model_dim, model_dim, num_heads)
        self.ffn              = FFN(model_dim)

    def forward(self, encoded_src, trg, mask):
        encoded_trg = addNorm(trg, self.self_attention(trg,trg,trg,mask))
        ffn_in      = addNorm(encoded_trg, self.encdec_attention(encoded_src,encoded_src,encoded_trg,None))
        out         = addNorm(ffn_in, self.ffn(ffn_in))
        return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, N=1):
        super(type(self),self).__init__()
        self.embedding = Embedding(vocab_size,model_dim)
        self.layers    = nn.ModuleList((DecoderLayer(model_dim, num_heads) for _ in range(N)))

    def forward(self, encoded_src, trg, mask):
        out = self.embedding(trg)
        for layer in self.layers:
            out = layer(encoded_src, out, mask)
        return out

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
                 model_dim, num_heads,
                 N=1):
        super(type(self),self).__init__()
        self.encoder    = Encoder(input_vocab_size, model_dim, num_heads, N)
        self.decoder    = Decoder(output_vocab_size, model_dim, num_heads, N)
        self.classifier = nn.Linear(model_dim,output_vocab_size)

    def forward(self, src, trg, trg_mask):
        enc_src = self.encoder(src)
        enc_trg = self.decoder(enc_src, trg, trg_mask)
        return F.softmax(self.classifier(enc_trg), dim=-1)


if __name__ == '__main__':
    k_dim, v_dim, n_h, seq_l = 4, 2, 3, 6
    mha = MultiHeadAttention(k_dim, v_dim, 1)
    q, k, v = torch.ones(1, k_dim, seq_l),  torch.ones(1, k_dim, seq_l),  torch.ones(1, v_dim, seq_l)
    out = mha(q,k,v)
    print(q.shape, out.shape)

    from attention import make_output_mask

    x = torch.ones(1,20).long()
    m = make_output_mask(x.shape[-1])[2:3]
    print(m)
    trans = Transformer(1000, 24, 64, 4, 2)
    out = trans(x,x,m)
    print(x.size(),  out.size())
    print(out)
    print(m)
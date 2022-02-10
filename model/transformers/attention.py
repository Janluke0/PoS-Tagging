import torch


class Attention(torch.nn.Module):
    """Multiheaded attention module.

    Args:
        input_size (int): number of input features.
        output_size (int): number of output features.
        key_dim (int): size of key and query vectors.
        value_dim (int): value vectors.
        head (int): number of parallel attention heads.
    """
    def __init__(self, input_size, output_size, key_dim=64, value_dim=64, heads=8):
        super().__init__()
        self.heads    = heads
        self.linear_k = torch.nn.Conv1d(input_size,  key_dim   * heads, 1, bias=False)
        self.linear_v = torch.nn.Conv1d(input_size,  value_dim * heads, 1, bias=False)
        self.linear_q = torch.nn.Conv1d(output_size, key_dim   * heads, 1, bias=False)
        self.softmax  = torch.nn.Softmax(1)
        self.scaling  = key_dim ** -0.5
        
        self.linear   = torch.nn.Linear(value_dim * heads, output_size)

    def forward(self, x, y, mask=None):
        """Embed the input values according to the attention mechanism.

        Args:
            x (tensor BxIxT): input tensor.
            y (tensor BxOxS): output tensor.
            mask (tensor TxS): optional attention mask.

        Returns:
            A (BxExS) tensor representing the embedded features where
            E is the product between the number of heads and the size
            of value vectors.

        The input tensor represents a batch of sequences of T feature
        vectors, each one composed of I values.

        The output tensor represents a batch of sequences of S feature
        vectors, each one composed of O values.

        The optional mask can be used to enable or disable the
        contribution of input elements in the computation of output
        elements.  The mask is a TxS matrix with elements 0 (enabled)
        or -inf (disabled) values.

        """
        b = x.size(0)
        t = x.size(2)
        s = y.size(2)
        bh = b * self.heads
        # 1) Compute keyes, values and queries with position-wise
        # linear operations (convolutions with kernel of size one).
        keys = self.linear_k(x).view(bh, -1, t)   # BxIxT ==> (BH)xDxT
        values = self.linear_v(x).view(bh, -1, t)  # BxIxT ==> (BH)xDxT
        
        queries = self.linear_q(y).view(bh, -1, s)  # BxOxS ==> (BH)xDxS
        # 2) Compute attention coefficients with a scaled dot product.
        dot = torch.bmm(keys.transpose(1, 2), queries)  # ==> (BH)xTxS
        if mask is not None:
            dot = dot + mask[None, :, :]
        attention = self.softmax(self.scaling * dot)
        # 3) Aggregate the values weighted by the attention coeffcients.
        output = torch.bmm(values, attention)  # ==> (BH)xDxS
        output = output.view(b, -1, s)  # ==> BxExS
        return self.linear(output.transpose(1,2))


def make_output_mask(length, device=None):
    idx = torch.tril_indices(length, length, -1)
    m = torch.zeros(length, length, device=device)
    m.data[idx[0], idx[1]] = -float("inf")
    return m


def positional_encoding(length, dim, scale, device=None):
    """Positional encoding.

    length: output length
    dim: encoding size
    scale: maximum scale

    return: a [length x dim] float tensor
    """
    p = torch.arange(length, device=device)
    adim = torch.arange(dim, device=device)
    i = torch.div(adim, 2, rounding_mode='trunc') * 2 #(adim // 2) * 2
    phase = (adim % 2) * (3.14159265359 / 2)
    freq = scale ** (i / dim)
    angle = (p / freq.unsqueeze(1)) + phase.unsqueeze(1)
    return torch.sin(angle)


class PositionalEncoding1d(torch.nn.Module):
    def __init__(self, scale=10000):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        p = positional_encoding(x.size(2), x.size(1), self.scale, device=x.device)
        return x + p[None, :, :]


class PositionalEncoding2d(torch.nn.Module):
    def __init__(self, scale=10000):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        py = positional_encoding(x.size(2), x.size(1), self.scale, device=x.device)
        px = positional_encoding(x.size(3), x.size(1), self.scale, device=x.device)
        return x + py[None, :, :, None] + px[None, :, None, :]


if __name__ == "__main__":
    attention = Attention(32, 40)
    x = torch.ones(3, 32, 10)
    y = torch.ones(3, 40, 7)
    out = attention(x, y)
    print(x.size(), y.size(), out.size())
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(torch.nn.Module):
    def __init__(self, n_token, d_embed):
        super(Embedding, self).__init__()
        self.embedding_size = d_embed
        self.vocabulary_size = n_token

        self.embedding_w = torch.nn.Parameter(torch.Tensor(self.embedding_size, self.vocabulary_size))
        torch.nn.init.uniform_(self.embedding_w, a=-0.1, b=0.1)
        #torch.nn.init.normal_(self.embedding_w)
        self.embedding_b = torch.nn.Parameter(torch.Tensor(1, self.embedding_size))
        self.embedding_b.data.fill_(0)

    def forward(self, x):
        return F.linear(x, self.embedding_w, self.embedding_b)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class TimePositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(TimePoisiotnalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        pass



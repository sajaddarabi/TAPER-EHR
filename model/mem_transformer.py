import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

# append transformer_utils
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(path, 'transformer_utils'))
from multiheadattn import MultiHeadAttn
from embedding import Embedding, PositionalEmbedding
from positionwise_FF import PositionwiseFF
from init_weights import weights_init

__all__ = ['MemTransformerLM']

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class MemTransformerLM(BaseModel):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, clamp_len=-1,
                 sample_softmax=-1, demographics_len=0):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = Embedding(n_token, d_embed)
        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len

        self.clamp_len = clamp_len
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(
                DecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm)
                )


        self.pos_emb = PositionalEmbedding(self.d_model)
        self.loss = nn.BCEWithLogitsLoss()
        self.demographics_len = demographics_len
        self.fc = nn.Linear(self.d_embed + self.demographics_len, self.n_token, bias=True)
        weights_init(self)

    def _forward(self, dec_inp):
        qlen, bsz, _ = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        klen = qlen
        # decoder attention mask
        dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1).byte()[:,:,None]

        hids = []
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb + pos_emb[-qlen:])

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            core_out = layer(core_out, dec_attn_mask=dec_attn_mask.bool())
            hids.append(core_out)

        core_out = self.drop(core_out)

        return core_out

    def forward(self, data, target, **kwargs):
        target_mask = kwargs.get('target_mask', None)
        demo = kwargs.get('demo', None)
        tgt_len = target.size(0)
        hidden = self._forward(data)

        if (target_mask is not None):
            target_mask = target_mask.unsqueeze(2)
            hidden = torch.mul(hidden, target_mask)
        pred_hid = hidden[-tgt_len:]

        if (demo is not None and self.demographics_len):
            pred_hid = torch.cat((pred_hid, demo), dim=2)

        pred_hid = pred_hid.transpose(1, 0).contiguous().view(-1, pred_hid.size(-1))
        logits = self.fc(pred_hid)
        return [logits, self.word_emb.embedding_w]

    def get_embedding(self):
        return self.word_emb

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters if p is not None])
        return '\nTrainable parameters: {}'.format(params)

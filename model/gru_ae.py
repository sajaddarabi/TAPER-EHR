from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from base import BaseModel
import math
from .gelu import GeLU

from .transformer_utils.multiheadattn import MultiHeadAttn

MAX_LENGTH = 30
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.multiheadattn = MultiHeadAttn(8, self.hidden_size, 128, dropout=0.1)
        self.gru = nn.GRU(self.input_size, self.hidden_size)

    def forward(self, input, hidden, i_s, b_is):
        if (len(input.shape) == 2):
            input = input.unsqueeze(1)

        output = input
        output, hidden = self.gru(output, hidden)
        output = self.multiheadattn(output)
        output = output[i_s, b_is, :]
        output = F.relu(output)

        if (len(output.shape) == 1):
            output = output.unsqueeze(0).unsqueeze(1)
        elif (len(output.shape) == 2):
            output = output.unsqueeze(0)
        return output

    def set_device(self, device):
        self.device = device


    def init_hidden(self, batch_size, device=None):
        self.batch_size = batch_size
        if (hasattr(self, 'device')):
            device = self.device

        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def init_from_state_dict(self, state_dict):
        td = {k:v for k, v in self.named_parameters() if 'encoder.' + k in state_dict}
        self.load_state_dict(td)

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, input_size)

    def forward(self, input, hidden):
        if len(input.shape) == 2:
            input = input.unsqueeze(1)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
            input = input.unsqueeze(1)
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

        return output, hidden

    def set_device(self, device):
        self.device = device


    def init_hidden(self, batch_size, device=None):
        self.batch_size = batch_size
        if (hasattr(self, 'device')):
            device = self.device

        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, **kwargs):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.multiheadattn = MultiHeadAttn(8, self.hidden_size, 128, dropout=0.1)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):#encoder_outputs, hidden,  i_s, b_is):
        outputs = torch.zeros((self.max_length, self.batch_size, self.output_size)).to(self.device)
        output = input
        for i in range(input.shape[0]):
            output, hidden = self.gru(output, hidden)
            o = self.out(output[0])
            outputs[i, :, :] = o
        return outputs

    def set_device(self, device):
        self.device = device

    def init_hidden(self, batch_size, device=None):
        self.batch_size = batch_size
        if (hasattr(self, 'device')):
            device = self.device
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class GRUAE(BaseModel):
    def __init__(self, input_size, hidden_size, teacher_forcing_ratio, **kwargs):
        super(GRUAE, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = EncoderRNN(input_size, hidden_size, **kwargs)
        self.decoder = AttnDecoderRNN(hidden_size, input_size, **kwargs)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, x):
        pass

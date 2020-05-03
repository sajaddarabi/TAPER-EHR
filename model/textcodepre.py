import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bert_things.pytorch_pretrained_bert import BertConfig, BertModel, BertPreTrainedModel
from base import BaseModel
from model.mem_transformer import MemTransformerLM
from model.gru_ae import *
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.input_size, self.hidden_size)

    def forward(self, input, hidden):
        if (len(input.shape) == 2):
            input = input.unsqueeze(1)
        output = input
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def set_device(self, device):
        self.device = device

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def init_from_state_dict(self, state_dict):
        td = {k:v for k, v in self.named_parameters() if 'encoder.' + k in state_dict}
        self.load_state_dict(td)

class TextCodePre(BaseModel):
    def __init__(self, transformer_state_path, text_summarizer_state_path, num_classes, demographics_size=0, text=True, codes=True, div_factor=2, dropout=0.5):
        super(TextCodePre, self).__init__()

        self.num_classes = num_classes
        self.codes = codes
        self.text = text
        self.demographics_size = demographics_size

        state_dict = torch.load(transformer_state_path)
        transformer_config = state_dict['config']
        state_dict = state_dict['state_dict']

        transformer_args = transformer_config['model']['args']
        self.transformer = MemTransformerLM(**transformer_args)

        self.transformer.load_state_dict(state_dict)
        self.transformer.eval()

        state_dict = torch.load(text_summarizer_state_path)
        summ_config = state_dict['config']
        state_dict = state_dict['state_dict']

        summ_args = summ_config['model']['args']

        self.summarizer = GRUAE(**summ_args)
        self.summarizer.load_state_dict(state_dict)


        self.patient_representation_size = int(self.text) * self.summarizer.hidden_size + self.transformer.d_embed * int(self.codes) + self.demographics_size
        self.predictor = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(self.patient_representation_size, self.patient_representation_size// div_factor),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.patient_representation_size // div_factor, self.num_classes))

    def forward(self, x, device='cuda'):
        x_codes, x_cl, x_text, x_tl, b_is, demo = x

        x_codes = x_codes.to(device)
        x_cl = x_cl.to(device)
        x_text = x_text.to(device)
        x_tl = x_tl.to(device)
        demo = demo.to(device)
        b_is = b_is.to(device)
        batch_size = b_is.shape[0]

        if (self.text):
            with torch.no_grad():
                hidden = self.summarizer.encoder.init_hidden(batch_size, device)
                enc = self.summarizer.encoder(x_text, hidden, x_tl, b_is) #sentence summarizer
                enc = enc.squeeze()

            # TODO: some sentences are very short... include more data?
            #enc = enc[x_tl, b_is, :] # simply take the last sentence

        #TODO: how to interpret transformer hidden output

        #x_code = x_code.unsqueeze(1) # only needed if feeding single row and len(shape) == 3
        if (self.codes):
            with torch.no_grad():
                mem_out = self.transformer._forward(x_codes)
                mem_out = mem_out[x_cl, b_is, :]

        #mem_out = torch.mean(mem_out, dim=0)
        patient_representation = torch.tensor([], device=device)

        if (self.codes and self.text):
            patient_representation = torch.cat((enc, mem_out), dim=1)
        elif (self.codes):
            patient_representation = mem_out
        elif (self.text):
            patient_representation = enc
        if (self.demographics_size > 0):
            if (len(patient_representation.shape) == 0):
                patient_representation = demo
            else:
                patient_representation = torch.cat((patient_representation, demo), dim=1)

        logits = self.predictor(patient_representation)
        if self.num_classes > 1:
            log_probs = F.log_softmax(logits, dim=1).squeeze()
        else:
            log_probs = torch.sigmoid(logits).squeeze()
        return log_probs, logits.squeeze()

    def __str__(self):
            """
            Model prints with number of trainable parameters
            """
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters if p is not None])
            return '\nTrainable parameters: {}'.format(params)

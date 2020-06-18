import torch
import torch.nn as nn
import torch.nn.functional as F
from model.bert_things.pytorch_pretrained_bert import BertConfig, BertModel, BertPreTrainedModel
import numpy as np

class BertTextModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertTextModel, self).__init__(config)
        self.bert = BertModel(config)

    def forward(self, data, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.bert(data, attention_mask=attention_mask, output_all_encoded_layers=output_all_encoded_layers)

        return sequence_output, pooled_output

    def init_from_state_dict(self, state_dict):
        td = {k:v for k, v in self.named_parameters() if k in state_dict}
        self.load_state_dict(td)


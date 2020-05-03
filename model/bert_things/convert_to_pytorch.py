import os
import argparse
import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from pytorch_pretrained_bert import BertConfig, BertForPreTraining

parser = argparse.ArgumentParser('convert biobert tf checkpoint to pytorch checkpoint ')

# NOTE: to properly load checkpoints path must be the path to the tf ckpt file including filename before .data
parser.add_argument('--path', default='.', type=str, help='path to tensorflow checkpoint')
parser.add_argument('--config', default='./bert_config.json', type=str, help='path to bert config file')
parser.add_argument('--save', default='./pytorch_weights', type=str, help='path to save file')

args = parser.parse_args()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    excluded = ['BERTAdam','_power','global_step']
    init_vars = list(filter(lambda x:all([True if e not in x[0] else False for e in excluded]),init_vars))
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
if __name__ == '__main__':
    convert_tf_checkpoint_to_pytorch(args.path, args.config, os.path.abspath(args.save))


try:
    import colored_traceback.always
except:
    pass
try:
    import nni
except:
    pass
import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.mem_transformer as module_arch
from tqdm import tqdm
from utils import Logger
import pickle
import numpy as np

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def import_module(name, config):
    return getattr(__import__("{}.{}".format(name, config[name]['module_name'])), config[name]['type'])

def mod_config(config, nni_params):
    if (nni_params == None):
        return config
    def recurse_dict(d, k, v):
        if (k in d):
            d[k] = v
            return d
        for kk, vv in d.items():
            if (type(vv) == dict):
                d[kk] = recurse_dict(vv, k, v)
        return d

    for k, v in nni_params.items():

        if k in config:
            config[k] = v
            continue
        for kk, vv in config.items():
            if (type(vv) == dict):
                config[kk] = recurse_dict(vv, k, v)
    return config


def main(config, resume, nni_params={}):

    config = mod_config(config, nni_params)
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = import_module('model', config)(**config['model']['args'])
    #model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    Trainer = import_module('trainer', config)
    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structmed Trainer')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    #if args.device:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    #torch.set_default_tensor_type(torch.cuda.FloatTensor if args.device else torch.FloatTensor)
    params = {}
    try:
        params = nni.get_next_parameter()
    except:
        pass
    #params = {"text": False}
    #params = {"text": True, "codes": False, "learning_rate": 0.0001, "demographics_size": 0, "batch_size": 16, "div_factor": 1, "step_size": 40, "class_weight_1": 4.616655939419362, "class_weight_0": 0.81750651640358}
    main(config, args.resume, params)

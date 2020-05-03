import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from train import get_instance
from train import import_module

import pickle

def main(config, resume):
    # setup data_loader instances

    data_loader = get_instance(module_data, 'data_loader', config)
    #data_loader = getattr(module_data, config['data_loader']['type'])(
    #    config['data_loader']['args']
    #    batch_size=512,
    #    shuffle=False,
    #    validation_split=0.0,
    #    training=False,
    #    num_workers=2
    #)

    data_loader = data_loader.split_validation()

    # build model architecture

    model = import_module('model', config)(**config['model']['args'])
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    predictions = {"output": [], "target": []}


    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            #data, target = data.to(device), target.to(device)
            target = target.to(device)
            output = model(data, device)
            #
            # save sample images, or do something with output here
            #

            output, logits = output
            predictions['output'].append(output.cpu().numpy())
            predictions['target'].append(target.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)
    save_dir = os.path.join(os.path.abspath(os.path.join(resume, '..', '..')))
    predictions['output'] = np.hstack(predictions['output'])
    predictions['target'] = np.hstack(predictions['target'])
    print(save_dir + '/predictions.pkl')
    with open(os.path.join(save_dir, 'predictions.pkl'), 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume)

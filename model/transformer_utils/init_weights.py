import torch.nn as nn
def init_weight(weight, config={}):
    if config['init'] == 'uniform':
        nn.init.uniform_(weight, -config['init_range'], config['init_range'])
    elif config['init'] == 'normal':
        nn.init.normal_(weight, 0.0, config['init_std'])

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m, config={}):
    if config == {} or ('init' not in config):
        config['init'] = 'uniform'
        config['init_range'] = 0.1
        config['init_std'] = 0.02
        config['proj_init_std'] = 0.01
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, config)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, config['proj_init_std'])
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, config)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.weight, config)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, config['proj_init_std'])
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, config['init_std'])
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, config)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, config)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, config)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

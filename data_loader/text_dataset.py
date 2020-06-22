import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from data_loader.utils.vocab import Vocab
import random

class TextDataset(data.Dataset):
    def __init__(self, data_path, text, batch_size, train=True, split_num=1):
        super(TextDataset, self).__init__()
        self.root = data_path
        self.batch_size = batch_size
        self.train = train

        self.data = pickle.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
        self.data_info = self.data['info']
        self.data = self.data['data']
        data_split_path = os.path.join(data_path, 'splits', 'split_{}.pkl'.format(split_num)
        if (os.path.exists(data_split_path)):
            self.train_idx, self.valid_idx = pickle.load(open(data_split_path, 'rb'))
            self.train_data = self.create_dataset(text, self.train_idx)
            self.valid_data = self.create_dataset(text, self.valid_idx)
            self.train_idx = np.asarray(range(0, len(self.train_data)))
            self.valid_idx = len(self.train_idx) + np.asarray(range(0, len(self.valid_data)))
        else:
            self.train_data = self.create_dataset(text, self.data.keys())



    def create_dataset(self, text_type, keys):
        t = []
        for j, k in enumerate(keys):
            for i, v in enumerate(self.data[k]):
                l = v['text_{}_len'.format(text_type)] // 512 + (int(v['text_{}_len'.format(text_type)] % 512 > 0))
                if (l == 0):
                    continue
                tt = (v['text_embedding_{}'.format(text_type)], l)
                t.append(tt)
        return t

    def __getitem__(self, index):
        if (hasattr(self, 'train_idx') and index in self.train_idx) or
            (index < len(self.train_data)):
            d = self.train_data[index]
        else:
            d = self.valid_data[index - len(self.train_data)]

        x = self.process_data(d)
        return x

    def process_data(self, d):
        x = torch.tensor(d[0])
        i = torch.tensor([d[1] - 1])
        return x, i

    def __len__(self):
        return len(self.train_data)

def collate_fn(data):
    x_text, i_s = zip(*data)
    x_text = torch.stack(x_text, dim=1)
    i_s = torch.stack(i_s, dim=0)
    b_is = torch.arange(i_s.shape[0]).reshape(tuple(i_s.shape))
    return x_text, i_s.squeeze(), b_is.squeeze()

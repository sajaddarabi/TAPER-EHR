import torch
import torch.utils.data as data
import os
import pickle
import numpy as np

class SeqCodeDataset(data.Dataset):
    def __init__(self, data_path, batch_size, train=True, med=False, diag=False, proc=False, cptcode=False, split_num=1):
        self.proc = proc
        self.med = med
        self.diag = diag
        self.cpt= cptcode

        self.train = train
        self.batch_size = batch_size

        self.data = pickle.load(open(os.path.join(data_path, 'data.pkl'), 'rb'))
        self.data_info = self.data['info']
        self.data = self.data['data']


        data_split_path = os.path.join(self.data_path, 'splits', 'split_{}.pkl'.format(split_num))
        if (os.path.exists(data_split_path)):
            self.train_idx, self.valid_idx = pickle.load(open(data_split_path, 'rb'))

        self.diag_vocab = pickle.load(open(os.path.join(data_path, 'diag_vocab.pkl'), 'rb'))
        self.med_vocab = pickle.load(open(os.path.join(data_path, 'med_vocab.pkl'), 'rb'))
        self.cpt_vocab = pickle.load(open(os.path.join(data_path, 'cpt_vocab.pkl'), 'rb'))
        self.proc_vocab = pickle.load(open(os.path.join(data_path, 'proc_vocab.pkl'), 'rb'))

        self.keys = list(self.data.keys())
        self.keys = self._get_keys()

        self.max_len = self._findmax_len(self.keys)

        self.num_dcodes = len(self.diag_vocab)
        self.num_pcodes = len(self.proc_vocab)
        self.num_mcodes = len(self.med_vocab)
        self.num_ccodes = len(self.cpt_vocab)

        self.num_codes = self.diag * self.num_dcodes + \
                         self.cpt * self.num_ccodes + \
                         self.proc * self.num_pcodes + \
                         self.med * self.num_mcodes

        self.demographics_shape = self.data_info['demographics_shape']

    def _gen_idx(self, keys, min_adm=2):
        idx = []
        for k in keys:
            v = self.data[k]
            if (len(v) < min_adm):
                continue
            for i, vv in enumerate(v):
                idx.append((k, i))
        return idx

    def _get_keys(self, min_adm=2):
        keys = []
        for k,v in self.data.items():
            if len(v) < min_adm:
                continue
            keys.append(k)
        return keys

    def _find_num_tokens(self, ext='D'):
        nc = 0
        for k in self.vocab.sym2idx.keys():
            t = k.split('_')
            tt = t[0]
            if (tt == ext):
                nc += 1
        return nc

    def _findmax_len(self, keys):
        m = 0
        for k,v in self.data.items():
            if (len(v) > m):
                m = len(v)
        return m

    def __len__(self):
        if self.train:
            return len(self.keys)
        else:
            return 0

    def __getitem__(self, k):
        x = self.preprocess(self.data[k])
        return x#, ivec, jvec

    def preprocess(self, seq):
        """ create one hot vector of idx in seq, with length self.num_codes

            Args:
                seq: list of ideces where code should be 1

            Returns:
                x: one hot vector
                ivec: vector for learning code representation
                jvec: vector for learning code representation
        """

        x = torch.zeros((self.num_codes, self.max_len ), dtype=torch.long)
        d = torch.zeros((self.demographics_shape, self.max_len), dtype=torch.float)
        mask = torch.zeros((self.max_len, ), dtype=torch.float)
        ivec = []
        jvec = []

        for i, s in enumerate(seq):
            dcode = list(s['diagnoses']) * self.diag
            pcode = list(len(self.diag_vocab) * self.diag + np.asarray(s['procedures'])) * self.proc
            mcode = list(len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + np.asarray(s['medications'])) * self.med
            cptcode = list(len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + len(self.cpt_vocab) * self.cpt + np.asarray(s['cptproc'])) * self.cpt
            demo = s['demographics']
            ss = (dcode) + (pcode) + (mcode) + (cptcode)

            for j in ss:
                for k in ss:
                    if j == k:
                        continue
                ivec.append(j)#torch.cat(ivec, i)
                jvec.append(k)#torch.cat(jvec, j)
            x[ss, i] = 1
            d[:, i] = torch.Tensor(demo)
            mask[i] = 1

        return x.t(), mask, torch.LongTensor(ivec), torch.LongTensor(jvec), d.t()

def collate_fn(data):
    """ Creates mini-batch from x, ivec, jvec tensors

    We should build custom collate_fn, as the ivec, and jvec have varying lengths. These should be appended
    in row form

    Args:
        data: list of tuples contianing (x, ivec, jvec)

    Returns:
        x: one hot encoded vectors stacked vertically
        ivec: long vector
        jvec: long vector
    """
    x, m, ivec, jvec, demo = zip(*data)
    m = torch.stack(m, dim=1)
    x = torch.stack(x, dim=1)
    ivec = torch.cat(ivec, dim=0)
    jvec = torch.cat(jvec, dim=0)
    demo = torch.stack(demo, dim=1)
    return x, m, ivec, jvec, demo

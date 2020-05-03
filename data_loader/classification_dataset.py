import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
from data_loader.utils.vocab import Vocab
from sklearn.model_selection import train_test_split
import random

class ClassificationDataset(data.Dataset):
    def __init__(self, data_path, text, batch_size, y_label='los', train=True, transform=None, target_transform=None, download=False, balanced_data=False, validation_split=0.0, sequential=True, split_num=1, med=False, diag=False, proc=False, cptcode=False, sg_text=False, sg_path='', min_adm=1):
        super(ClassificationDataset).__init__()
        self.proc = proc
        self.med = med
        self.diag = diag
        self.cpt = cptcode
        self.text = text
        self.sg_text = sg_text
        self.sg_path = sg_path

        self.data_path = data_path
        self.batch_size = batch_size
        self.train = train
        self.y_label = y_label
        self.validation_split = validation_split
        self.balanced_data = balanced_data

        self.pre = self.preprocess
        if (not sequential):
            self.pre = self.preprocess_non_seq

        self.data = pickle.load(open(os.path.join(self.data_path, 'data_filt.pkl'), 'rb'))
        self.data_info = self.data['info']
        self.data = self.data['data']

        self.train_idx, self.valid_idx = pickle.load(open(os.path.join(self.data_path, 'splits', 'split_{}.pkl'.format(split_num)), 'rb'))
        self.train_idx = self._filt_indices(self.train_idx, min_adm)
        self.valid_idx = self._filt_indices(self.valid_idx, min_adm)

        self.text_dim = self._get_text_dim(self.data)
        self.demographics_shape = self.data_info['demographics_shape']

        self.diag_vocab = pickle.load(open(os.path.join(data_path, 'diag_vocab.pkl'), 'rb'))
        self.med_vocab = pickle.load(open(os.path.join(data_path, 'med_vocab.pkl'), 'rb'))
        self.cpt_vocab = pickle.load(open(os.path.join(data_path, 'cpt_vocab.pkl'), 'rb'))
        self.proc_vocab = pickle.load(open(os.path.join(data_path, 'proc_vocab.pkl'), 'rb'))

        self.keys = list(map(int, self.data.keys()))
        self.max_len = self._findmax_len(self.keys)

        self.num_dcodes = len(self.diag_vocab)
        self.num_pcodes = len(self.proc_vocab)
        self.num_mcodes = len(self.med_vocab)
        self.num_ccodes = len(self.cpt_vocab)

        self.num_codes = self.diag * self.num_dcodes + \
                         self.cpt * self.num_ccodes + \
                         self.proc * self.num_pcodes + \
                         self.med * self.num_mcodes#len(self.vocab)


        self.train_indices = self._gen_indices(self.train_idx)
        self.valid_indices = self._gen_indices(self.valid_idx)

        self.train_idx = np.asarray(range(len(self.train_indices)))
        self.valid_idx = len(self.train_indices) + np.asarray(range(len(self.valid_indices)))




        if (self.balanced_data):
            self.train_idx = self._gen_balanced_indices(self.train_idx)
            #self.valid_idx = self._gen_balanced_indices(self.valid_idx)


    def _filt_indices(self, indices, min_adm):
        t = []
        for idx in indices:
            if (len(self.data[idx]) < min_adm):
                continue
            t.append(idx)
        return t

    def _gen_balanced_indices(self, indices):
        ind_idx = {}

        for idx in indices:
            label = self.get_label(idx)
            if (label not in ind_idx):
                ind_idx[label] = [idx]
            else:
                ind_idx[label].append(idx)

        tr = []
        te = []


        lens = sorted([len(v) for k, v in ind_idx.items()])

        if (len(lens) > 3):
            num_samples = lens[-2]
        else:
            num_samples = lens[0]

        for k, v in ind_idx.items():
            v = np.asarray(v)

            if (len(v) > num_samples):
                v = v[np.random.choice(np.arange(len(v)), num_samples)]

            #train, test = train_test_split(v, test_size=self.validation_split, random_state=1)
            #te.append(test)

            tr.append(v)

        train = np.concatenate(tr)
        #test = np.concatenate(te)
        return train#, test

    def _gen_indices(self, keys):
        indices = []
        for k in keys:
            v = self.data[k]
            (x_codes, x_cl, x_text, x_tl, demo, y) = self.preprocess([k, 0])
            if torch.sum(x_text):
                indices.append([k, 0])
            if (len(y.size()) == 0) or (len(demo.size()) == 0):
                import pdb; pdb.set_trace()
            for j in range(len(v)):
                if ((j + 1) == len(v)):
                   continue
                #(x_codes, x_cl, x_text, x_tl, demo, y) = self.preprocess([k, j+1])
                #if not (x_text == [] and torch.sum(x_codes) == 0):
                if (len(y.size()) == 0) or (len(demo.size()) == 0):
                    import pdb; pdb.set_trace()

                indices.append([k, j + 1])
        return indices

    def _findmax_len(self, keys):
        m = 0
        for k in keys:
            vv = self.data[k]
            if (len(vv) > m):
                m = len(vv)
        return m

    def __getitem__(self, index):
        if (index in self.train_idx):
            idx = self.train_indices[index]
        else:
            idx = self.valid_indices[index - len(self.train_indices)]
        x = self.pre(idx)
        return x

    def _get_text_dim(self, data):
        for k, v in data.items():
            for vv in v:
                x_text = torch.Tensor(vv['text_embedding_{}'.format(self.text)])
                if (len(x_text.size())):
                    return x_text.shape

    def preprocess(self, idx):
        seq = self.data[idx[0]]
        n = idx[1]
        x_codes = torch.zeros((self.num_codes, self.max_len), dtype=torch.float)
        demo = torch.Tensor(seq[n]['demographics'])
        x_text = torch.zeros(self.text_dim)
        for i in range(n):
            if(i + 1) == len(seq):
               continue
            s = seq[i]
            dcode = list(s['diagnoses'])  * self.diag
            pcode = list(len(self.diag_vocab) * self.diag + np.asarray(s['procedures'])) * self.proc
            mcode = list(len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + np.asarray(s['medications'])) * self.med
            cptcode = list(len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + len(self.cpt_vocab) * self.cpt + np.asarray(s['cptproc'])) * self.cpt
            codes = (dcode) + (pcode) + (mcode) + (cptcode)
            x_codes[codes, i] = 1

            # TODO: summarize all text before time step t ?
            # as it stands we only summarize text of current prediction time window.
            #t = v['text_embedding']
            #x_text[:, i] = t
            #x_tl[i] = v['text_len']

        x_cl = torch.Tensor([n,])
        x_tl = seq[n]['text_{}_len'.format(self.text)] // 512 + int((seq[n]['text_{}_len'.format(self.text)] % 512) == 0)
        x_tl = torch.Tensor([x_tl, ])
        x_tt = torch.Tensor(seq[n]['text_embedding_{}'.format(self.text)])

        if (len(x_tt.shape) != 0):
            x_text = x_tt

        if (self.y_label == 'los'):
            los = seq[n]['los']
            if los != los:
                los = 9
            y = torch.Tensor([los - 1])
        elif self.y_label == 'readmission':
            y = torch.Tensor([seq[n]['readmission']])
        else:
            y = torch.Tensor([seq[n]['mortality']])

        return (x_codes.t(), x_cl, x_text, x_tl, demo, y)

    def get_label(self, idx):
        if (idx in self.train_idx):
            idx = self.train_indices[idx]
        else:
            idx = self.valid_indices[idx- len(self.train_indices)]
        seq = self.data[idx[0]]
        n = idx[1]
        if (self.y_label == 'los'):
            los = seq[n]['los']
            if los != los:
                los = 9
            y = torch.Tensor([los - 1])
        elif self.y_label == 'readmission':
            y = torch.Tensor([seq[n]['readmission']])
        else:
            y = torch.Tensor([seq[n]['mortality']])
        y = y.item()
        return y

    def preprocess_non_seq(self, idx):
        seq = self.data[idx[0]]
        n = idx[1]
        x_codes = torch.zeros((self.num_codes, ), dtype=torch.float)
        demo = torch.Tensor(seq[n]['demographics'])
        v = seq[n]
        if (n > 1):
            s= seq[n - 1]
            dcode = s['diagnoses']  * self.diag
            pcode = (len(self.diag_vocab) * self.diag + np.asarray(s['procedures'])) * self.proc
            mcode = (len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + np.asarray(s['medications'])) * self.med
            cptcode = (len(self.diag_vocab) * self.diag + len(self.proc_vocab) * self.proc + len(self.cpt_vocab) * self.cpt + np.asarray(s['cptproc'])) * self.cpt
            codes = list(dcode) + list(pcode) + list(mcode) + list(cptcode)
            x_codes[codes] = 1

        if (self.sg_text):
            x_text = self.sg_model.get_sentence_vector(v['text_{}_raw'.format(self.text)])
            x_text = torch.tensor(x_text)
        else:
            x_text = torch.Tensor(seq[n]['text_embedding_{}'.format(self.text)])
        x_tl = seq[n]['text_{}_len'.format(self.text)]
        x_tl = torch.Tensor([x_tl, ])
        if (self.y_label == 'los'):
            los = seq[n]['los']
            if los != los:
                los = 9
            y = torch.Tensor([los - 1])
        elif self.y_label == 'readmission':
            y = torch.Tensor([seq[n]['readmission']])
        else:
            y = torch.Tensor([seq[n]['mortality']])
        return (x_codes, x_text, x_tl, demo, y)

    def __len__(self):
        l = 0
        if (self.train):
            l = len(self.train_idx)
        else:
            l = len(self.valid_idx)

        return l

def collate_fn(data):
    x_codes, x_cl,  x_text, x_tl, demo, y_code = zip(*data)
    x_codes = torch.stack(x_codes, dim=1)
    demo = torch.stack(demo, dim=0)
    y_code = torch.stack(y_code, dim=1).long()
    x_text = torch.stack(x_text, dim=1)
    x_cl = torch.stack(x_cl, dim=0).long()
    x_tl = torch.stack(x_tl, dim=0).long()
    b_is = torch.arange(x_cl.shape[0]).reshape(tuple(x_cl.shape)).long()
    return (x_codes, x_cl.squeeze(),  x_text, x_tl.squeeze(), b_is.squeeze(), demo), y_code.squeeze()

def non_seq_collate_fn(data):
    x_codes, x_text, x_tl, demo, y_code = zip(*data)
    x_codes = torch.stack(x_codes, dim=0)
    demo = torch.stack(demo, dim=0)
    y_code = torch.stack(y_code, dim=1).long()
    x_text = torch.stack(x_text, dim=0)
    x_tl = torch.stack(x_tl, dim=0).long()
    b_is = torch.arange(x_tl.shape[0]).reshape(tuple(x_tl.shape)).long()
    return (x_codes, x_text, x_tl.squeeze(), b_is, demo), y_code.squeeze()

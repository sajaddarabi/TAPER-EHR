import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import Sampler

from .imbalanced_sampler import ImbalancedSampler

class MySequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return (self.data_source[i] for i in range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate, seed=0):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed
        self.dataset = dataset


        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0 and not hasattr(self.dataset, 'valid_idx'):
            return None, None
        idx_full = np.arange(self.n_samples)
        np.random.seed(self.seed)

        # shuffle indexes only if shuffle is true
        # if order matters don't shuffle
        # added for med2vec dataset where order matters
        len_valid = int(self.n_samples * split)
        if (self.shuffle):
            if (hasattr(self.dataset, 'valid_idx')):
                valid_idx = self.dataset.valid_idx
                train_idx = self.dataset.train_idx
            else:
                valid_idx = idx_full[0:len_valid]

            train_sampler = SubsetRandomSampler(train_idx)
            # use the balanced dataset sampler if balanced_data is set
            # this option can be passed to the dataset class
            if (hasattr(self.dataset, 'balanced_data') and self.dataset.balanced_data):
                train_sampler = ImbalancedSampler(self.dataset, train_idx)

            valid_sampler = SubsetRandomSampler(valid_idx)

        else:
            num_intervals = len(idx_full) // len_valid
            rand_i = np.random.randint(0, num_intervals)
            valid_idx = idx_full[rand_i * len_valid: (rand_i + 1) * len_valid]
            train_idx = np.delete(idx_full, np.arange(rand_i * len_valid, (rand_i + 1) * len_valid))

            if (hasattr(self.dataset, 'valid_idx')):
                valid_idx = self.dataset.valid_idx
                train_idx = self.dataset.train_idx

            train_sampler = MySequentialSampler(train_idx)
            valid_sampler = MySequentialSampler(valid_idx)
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

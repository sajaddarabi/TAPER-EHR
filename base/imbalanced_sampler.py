import torch
import torch.utils.data
import torchvision

class ImbalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) \
                        if indices is None else indices
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        self.label_to_count = label_to_count

        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif hasattr(dataset, 'get_label'):
            return dataset.get_label(idx)
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial (self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

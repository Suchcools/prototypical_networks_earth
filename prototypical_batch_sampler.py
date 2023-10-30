# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        # labels：数据集的标签，包含了所有样本的标签。
        # classes_per_it：每次迭代（episode）中随机选择的类别数量。
        # num_samples：每个类别采样的样本数量（包括支持集和查询集）。
        # iterations：迭代的次数（即epoch中采样的次数）。

        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''

        spc = self.sample_per_class # 每个类别采样的样本数量，包括支持集和查询集。
        cpi = self.classes_per_it # 每个iteration中随机选择的类别数量。

        for it in range(self.iterations):
            batch_size = spc * cpi   # 每个iteration采样的总样本数量。
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]  # 随机选择cpi个类别。
            for i, c in enumerate(self.classes[c_idxs]): # 对每个选中的类别进行采样。
                s = slice(i * spc, (i + 1) * spc) # 支持集和查询集的slice
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item() # 找到当前选中类别在索引中的位置。
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc] # 随机选择spc个样本作为当前类别的支持集和查询集。
                batch[s] = self.indexes[label_idx][sample_idxs] # 将选择的样本索引加入到采样的tensor中。
            batch = batch[torch.randperm(len(batch))] #打乱采样索引。
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

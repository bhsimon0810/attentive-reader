import pickle
from tqdm import tqdm
import numpy as np


class Dataset(object):
    def __init__(self, file_name):
        self.dataset = self._load_data(file_name)

    def _load_data(self, file_name):
        print("loading dataset from %s" % file_name)
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def _normalize(self, seqs):
        lengths = [len(seq) for seq in seqs]
        n_samples = len(seqs)
        max_len = np.max(lengths)
        x = np.zeros((n_samples, max_len)).astype(int)
        # x_mask = np.zeros((n_samples, max_len)).astype(float)
        for idx, seq in enumerate(seqs):
            # 0 for padding token
            x[idx, :lengths[idx]] = seq
            # x_mask[idx, :lengths[idx]] = 1.0
        return x, lengths

    def _batch_iter(self, dataset, minibatch_size, desc, shuffle=False):
        x1, x2, mask, y = dataset
        assert len(x1) == len(x2) == len(mask) == len(y)
        n = len(x1)
        idx_list = np.arange(0, n, minibatch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))

        for minibatch in tqdm(minibatches, desc):
            mb_x1 = [x1[t] for t in minibatch]
            mb_x2 = [x2[t] for t in minibatch]
            mb_mask = mask[minibatch]
            mb_y = [y[t] for t in minibatch]
            mb_x1, mb_x1_lengths = self._normalize(mb_x1)
            mb_x2, mb_x2_lengths = self._normalize(mb_x2)
            yield (mb_x1, mb_x1_lengths, mb_x2, mb_x2_lengths, mb_mask, mb_y)

    def batch_iter(self, minibatch_size, desc=None, shuffle=False):
        return self._batch_iter(self.dataset, minibatch_size, desc, shuffle)

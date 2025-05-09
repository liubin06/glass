import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_train(data_path):
    '''
    :param ：train data path
    :return: ndarray
    '''
    data = pd.read_csv(data_path,
                       header=None,
                       sep=',',
                       encoding='utf-8')
    print('Number of training samples: {}'.format(data.shape[0]))
    return np.array(data)


def load_validate(data_path):
    '''
    :param ：validate data path
    :return: ndarray
    '''
    data = pd.read_csv(data_path,
                       header=None,
                       sep=',',
                       encoding='utf-8')
    print('Number of validating samples: {}'.format(data.shape[0]))
    return np.array(data)


def load_test(data_path):
    '''
    :param ：validate data path
    :return: ndarray
    '''
    data = pd.read_csv(data_path,
                       header=None,
                       sep=',',
                       encoding='utf-8')
    return np.array(data)


class MyData(Dataset):
    def __init__(self, data, mean, std, input_dim, interval, noise_std, phase):
        '''
        :param data: ndarray with shape (n_samples, n_features+1 property)
        :param input_dim: number of input features (components)
        :param train: boolean indicating for loading training set or validation set
        :param std: the standard deviation of the noise for data augmentation
        :param interval: the interval of glass transition temperatures to be selected
        '''
        self.input_dim = input_dim
        self.data = torch.tensor(data[:, :self.input_dim], dtype=torch.float32)
        self.noise_std = noise_std
        self.phase = phase
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        if self.phase != 'Screening':
            self.label = torch.tensor(data[:, self.input_dim], dtype=torch.float32)
            self.binary_label = torch.tensor(
                [interval[0] <= self.label[id] <= interval[1] for id in range(len(self))]).float()

            self.target_id = [id for id in range(len(self)) if self.binary_label[id] == 1.]
            self.nontarget_id = [id for id in range(len(self)) if self.binary_label[id] == 0.]

    def __getitem__(self, idx):
        if self.phase == 'Training':
            feature = self.data[idx]
            label = self.binary_label[idx]
            if label == 1.:
                pos = self.data[np.random.choice(self.target_id)]
                neg = self.data[np.random.choice(self.nontarget_id)]
            else:
                pos = self.data[np.random.choice(self.nontarget_id)]
                neg = self.data[np.random.choice(self.target_id)]
            feature, pos, neg = self.perbulation(1, self.noise_std, feature), self.perbulation(1, self.noise_std,
                                                                                               pos), self.perbulation(1,
                                                                                                                      self.noise_std,
                                                                                                                      neg)
            return self.normalize(feature), self.normalize(pos), self.normalize(neg)

        elif self.phase == 'Evaluation':
            feature = self.data[idx]
            label = self.binary_label[idx]
            return self.normalize(feature), label

        elif self.phase == 'Screening':
            feature = self.data[idx]
            return self.normalize(feature)

    def __len__(self):
        return len(self.data)

    def perbulation(self, miu, sigma, feature):
        '''
        :param miu: mean value of the perbulation distribution
        :param sigma: standard deviation of the percolation distribution
        :param feature: input feature of dimension 19
        :return: augmentation feature
        '''
        perb = torch.normal(miu, sigma, [self.input_dim])
        augmentation = feature * perb

        return augmentation

    def normalize(self, feature):
        '''
        :param feature: input feature of dimension 19
        :return: normalized feature with summation of all components equal to 1
        '''
        normalized_feature = (feature - self.mean) / self.std
        return normalized_feature




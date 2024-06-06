import copy
import math

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from opacus import PrivacyEngine
import numpy as np
from random import randint
from opacus import PrivacyEngine
from src.test import tes_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class ModelUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, max_train_data_len=None ,min_train_data_len = None,dataset_all_len = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # args.local_bs=10，shuffle=True表示每次迭代训练时将数据洗牌
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True)
        # self.ldr_train = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        self.idxs = idxs
        self.len_dataset = len(dataset)
        self.max_train_data_len = max_train_data_len
        self.min_train_data_len = min_train_data_len

    def train(self, local_net, net):

        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)


        epoch_loss = []

        if self.args.sys_homo:
            local_ep = self.args.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le)


        for iter in range(local_ep):
            batch_loss = []
            grad_all = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()  # 清空梯度
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                loss.backward()  # 求梯度

                optimizer.step()  # 更新参数


                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class L1_weight_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_weight_Loss, self).__init__()
        self.gamma = gamma
    def dis(self, x, y):
        return torch.sum(torch.abs(x-y), dim=-1)
    def forward(self, x1, x2, train_set, train_weight, train_batch):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        dis_x1_x2 = self.dis(x1_train, x2_train)
        #a = F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg1))
        #print(a.size(),train_weight.size())
        #print((a*train_weight).size())
        loss11 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg1))*train_weight)
        loss12 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg2))*train_weight)
        loss21 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg1))*train_weight)
        loss22 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg2))*train_weight)
        loss = (loss11+loss12+loss21+loss22)/4
        return loss
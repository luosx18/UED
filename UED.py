# -*- coding: utf-8 -*-


import os
import argparse
import itertools

import apex
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from model import RAGA
from data import DBP15K, MedED
from loss import L1_Loss
from loss_weighted import L1_weight_Loss
from utils import add_inverse_rels, get_train_batch, get_train_batch_low_memory, get_hits, get_hits_stable, get_hits_hard, get_hits_stable_hard
#import ipdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=True)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--mt_pair", action="store_true", default=False)
    parser.add_argument("--merge_mt_pair", action="store_true", default=False)
    parser.add_argument("--mt_anchor_weight", action="store_true", default=False)
    parser.add_argument("--weight_decay", action="store_true", default=False)
    parser.add_argument("--coder", action="store_true", default=False)
    parser.add_argument("--loss2_weight", type=float, default=0.3)
    parser.add_argument("--loss2_only", action="store_true", default=False)
    parser.add_argument("--hard_test", action="store_true", default=False)
    args = parser.parse_args()
    return args


def init_data(args, device):
    if 'DBP15K' in args.data:
        data = DBP15K(args.data, args.lang, rate=args.rate,
                      random_init=args.random_init,
                      mt_pair=args.mt_pair,mt_anchor_weight=args.mt_anchor_weight)[0]
    elif 'MedED' in args.data:
        data = MedED(args.data, args.lang, rate=args.rate,
                      random_init=args.random_init,
                      mt_pair=args.mt_pair,mt_anchor_weight=args.mt_anchor_weight)[0]
    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)
    return data


def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    return x1, x2


def train(args, model, criterion, optimizer, data, train_batch, 
          criterion2=None, mt_anchor_batch=None, loss2_weight=None):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2)
    if args.merge_mt_pair:
        loss = criterion(x1, x2, data.merge_mt_pair_set, train_batch)
    elif args.mt_pair:
        loss = criterion(x1, x2, data.mt_pair_set, train_batch)
    elif args.loss2_only:
        loss = 0.0
    else:
        loss = criterion(x1, x2, data.train_set, train_batch)
    if args.mt_anchor_weight:
        anchor_pair = data.anchor_pair_list
        weight = data.weight_list.view(-1)
        loss2 = criterion2(x1, x2, anchor_pair, weight, mt_anchor_batch)#
        loss += loss2_weight * loss2
    optimizer.zero_grad()
    if args.cuda == False:
        loss.backward()
    else:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    optimizer.step()
    return loss


def test(args, model, data, stable=False,file=None):
    if args.hard_test:
        print('-'*16+'practical_setting'+'-'*16, file=file)
    else:
        print('-'*16+'relaxed_setting'+'-'*16, file=file)
    x1, x2 = get_emb(model, data)
    x1 = x1.cpu()
    x2 = x2.cpu()
    print('Train_set testing')
    print('-'*16+'Train_set'+'-'*16, file=file)
    if args.hard_test:
        get_hits_hard(x1, x2, data.train_set.cpu(),file=file)
    else:
        get_hits(x1, x2, data.train_set.cpu(),file=file)
    if args.mt_pair:
        print('mt_pair_set testing')
        print('-'*16+'mt_pair_set'+'-'*16,file=file)
        if args.hard_test:
            get_hits_hard(x1, x2, data.mt_pair_set.cpu(), file=file)
        else:
            get_hits(x1, x2, data.mt_pair_set.cpu(), file=file)
    print('Test_set testing')
    print('-'*16+'Test_set'+'-'*17,file=file)
    if args.hard_test:
        mean_H1, MRR = get_hits_hard(x1, x2, data.test_set.cpu(),file=file)
    else:
        mean_H1, MRR = get_hits(x1, x2, data.test_set.cpu(),file=file)
    if stable:#the DAA algorithm
        print('Test_set stable testing')
        if args.hard_test:
            get_hits_stable_hard(x1, x2, data.test_set.cpu(), file=file)
        else:
            get_hits_stable(x1, x2, data.test_set.cpu(), file=file)
    print('done!')
    print(file=file)
    return mean_H1, MRR 


def linear_weight(now_step,start,end,total_step):
    w = start-min(now_step,total_step)*(start-end)/total_step
    return w

def main(args):
    log_path = os.path.join(args.data, args.lang, 'train_log')
    if args.merge_mt_pair:
        log_path += 'merge_mt_pair'
    elif args.mt_pair:
        log_path += 'mt_pair'
    elif args.loss2_only:
        log_path += 'loss2_only'
    else:
        log_path += 'original'
    
    if args.mt_anchor_weight:
        log_path += '_mt_anchor_weight'
    
    if args.weight_decay:
        log_path += '_weight_decay'
    
    log_path = log_path + '_rate' + str(args.rate)
    file = open(log_path+'.txt','a+',encoding='utf-8')
    device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)
    
    if args.mt_pair:
        cleaned_mt_pair = []
        mt_1to2, mt_2to1 = {}, {}
        clean_1_set, clean_2_set = [], []
        for pair in data.mt_pair_set.cpu().numpy().tolist():
            if pair[0] not in mt_1to2:
                mt_1to2[pair[0]] = []
            mt_1to2[pair[0]].append(pair[1])
            if pair[1] not in mt_2to1:
                mt_2to1[pair[1]] = []
            mt_2to1[pair[1]].append(pair[0])
        
        for i in mt_1to2:
            if len(mt_1to2[i]) == 1:
                clean_1_set.append(i)
        
        for i in mt_2to1:
            if len(mt_2to1[i]) == 1:
                clean_2_set.append(i)
        
        clean_1_set = set(clean_1_set)
        clean_2_set = set(clean_2_set)
        for pair in data.mt_pair_set.cpu().numpy().tolist():
            if pair[0] in clean_1_set and pair[1] in clean_2_set:
                cleaned_mt_pair.append(pair)
        
        data.mt_pair_set = torch.tensor(cleaned_mt_pair).to(device)
    
    if args.merge_mt_pair:
        train_set = set()
        merge_mt_pair_set = []
        for pair in data.train_set.cpu().numpy().tolist():
            line = str(pair[0])+'\t'+str(pair[1])
            train_set.add(line)
            merge_mt_pair_set.append(pair)
        
        for pair in data.mt_pair_set.cpu().numpy().tolist():
            line = str(pair[0])+'\t'+str(pair[1])
            if line not in train_set:
                merge_mt_pair_set.append(pair)
        data.merge_mt_pair_set = torch.tensor(merge_mt_pair_set).to(device)
        print('merge_mt_pair_set size = ', data.merge_mt_pair_set.size())
    
    
    model = RAGA(data.x1.size(1), args.r_hidden).to(device)#adopt raga as the default graph embedding model
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), iter([data.x1, data.x2])))
    if args.cuda:
        model, optimizer = apex.amp.initialize(model, optimizer)
    criterion = L1_Loss(args.gamma)
    if args.mt_anchor_weight:
        criterion2 = L1_weight_Loss(args.gamma)
    else:
        criterion2 = None
    
    best_mean_H1 = 0
    mean_H1, MRR = test(args, model, data, args.stable_test, file)
    if best_mean_H1 < mean_H1:
        best_mean_H1 = mean_H1
        print('mean_H1=',mean_H1,file=file)
    
    for epoch in range(args.epoch):
        if epoch%args.neg_epoch == 0:
            x1, x2 = get_emb(model, data)
            print('get train_batch')
            if args.merge_mt_pair:
                train_batch = get_train_batch(x1, x2, data.merge_mt_pair_set, args.k)
                print('train_batch size =', train_batch.size())
            elif args.mt_pair:
                train_batch = get_train_batch(x1, x2, data.mt_pair_set, args.k)
                print('train_batch size =', train_batch.size())
            else:
                train_batch = get_train_batch(x1, x2, data.train_set, args.k)
            print('get train_batch done!')
            print('get mt_anchor_weight')
            if args.mt_anchor_weight:
                if data.anchor_pair_list.size()[0] > 20000*3:
                    mt_anchor_batch = get_train_batch_low_memory(x1, x2, data.anchor_pair_list, args.k)
                else:
                    mt_anchor_batch = get_train_batch(x1, x2, data.anchor_pair_list, args.k)
            else:
                mt_anchor_batch = None
            print('get mt_anchor_weight done!')
        if args.weight_decay:
            loss2_weight = linear_weight(epoch,args.loss2_weight,0,int(args.epoch/4))
        else:
            loss2_weight = args.loss2_weight
        print('loss2_weight =', loss2_weight, file=file)
        loss = train(args, model, criterion, optimizer, data, train_batch, criterion2, mt_anchor_batch, loss2_weight)
        print('Epoch:', epoch+1, '/', args.epoch, '\tLoss: %.3f'%loss, '\r', end='')
        print('Epoch:', epoch+1, '/', args.epoch, '\tLoss: %.3f'%loss, '\r', end='', file=file)
        if (epoch+1)%args.test_epoch == 0:
            mean_H1, MRR = test(args, model, data, args.stable_test, file)
            if best_mean_H1 < mean_H1:
                best_mean_H1 = mean_H1
                torch.save(model, log_path+'best.pkl')
                torch.save(data.x1, log_path+'best.pkl'+'x1')
                torch.save(data.x2, log_path+'best.pkl'+'x2')
    print('train done!', file=file)
    file.close()
    #x1, x2 = get_emb(model, data)
    #torch.save([x1[data.test_set[:, 0]].cpu(), x2[data.test_set[:, 1]].cpu()], 'x.pt')

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
    #device = 'cpu'
    #data = init_data(args, device).to(device)
    #ipdb.set_trace()
    #model = torch.load(args.data+log_path+'my.pkl')
    #model.eval()
    
    

# -*- coding: utf-8 -*-

import argparse
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import DBP15K, MedED
from loss import L1_Loss
from loss_weighted import L1_weight_Loss
from utils import add_inverse_rels, get_train_batch, get_train_batch_low_memory, get_hits, get_hits_stable, dangling_eval, get_hits_hard, get_hits_stable_hard
from sklearn.metrics import f1_score, precision_score, recall_score
from mip import Model, xsum, minimize, maximize, BINARY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/MedED")
    parser.add_argument("--lang", default="fr_en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rate", type=float, default=0.3)
    parser.add_argument("--r_hidden", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--stable_test", action="store_true", default=True)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument("--k_init", action="store_true", default=False)
    parser.add_argument("--mt_pair", action="store_true", default=True)
    parser.add_argument("--mt_anchor_weight", action="store_true", default=False)
    parser.add_argument("--weight_decay", action="store_true", default=False)
    parser.add_argument("--log_path", type=str)
    parser.add_argument("--reverse", action="store_true", default=False)
    parser.add_argument("--relaxed", action="store_true", default=False)
    parser.add_argument("--csls", action="store_true", default=False)
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





def extract_col_constraint_list(S, S_topk_row, top_id_row, S_empty_row,
                                S_topk_col, top_id_col, S_empty_col, union = True):
    top_id = [set(row.numpy().tolist()) for row in top_id_row]
    for i in range(len(top_id)):
        
        location = torch.nonzero(torch.where(top_id_col == i, torch.tensor(1), torch.tensor(0)))
        if union:
            top_id[i] = top_id[i].union(set(location[:,1].numpy().tolist()))
        else:
            top_id[i] = top_id[i].intersection(set(location[:,1].numpy().tolist()))
        top_id[i].add(S.size(1))
    
    for i in range(len(top_id)):
        top_id[i] = list(top_id[i])
        top_id[i].sort()
    
    top_id.append(torch.arange(S.size(1)).numpy().tolist())
    
    print('col_constraint generating')
    col_constraint = []
    for j in range(S.size(1)):
        temp = []
        for i in range(len(top_id)):
            if j in top_id[i]:
                for k in range(len(top_id[i])):
                    if j == top_id[i][k]:
                        temp.append([i,k])
        col_constraint.append(temp)
    
    c = [[] for i in range(S.size(0))]
    for i in range(S.size(0)):
        for j in top_id[i]:
            if j < S.size(1):
                c[i].append(S[i,j].numpy().tolist())
            elif j == S.size(1):
                c[i].append(S_empty_col[i,0].numpy().tolist())
    
    c = c+S_empty_row.numpy().tolist()
    return c, top_id, col_constraint


def quick_extract_col_constraint_list(S, S_topk_row, top_id_row, S_empty_row,
                                S_topk_col, top_id_col, S_empty_col, union = True):
    top_id = [set(row.numpy().tolist()) for row in top_id_row]
    for i in range(len(top_id)):
        
        location = torch.nonzero(torch.where(top_id_col == i, torch.tensor(1), torch.tensor(0)))
        if union:
            top_id[i] = top_id[i].union(set(location[:,1].numpy().tolist()))
        else:
            top_id[i] = top_id[i].intersection(set(location[:,1].numpy().tolist()))
        top_id[i].add(S.size(1))
    
    for i in range(len(top_id)):
        top_id[i] = list(top_id[i])
        top_id[i].sort()
    
    top_id.append(torch.arange(S.size(1)).numpy().tolist())
    
    print('col_constraint generating')
    time_start = time.time()
    seleted = torch.zeros(S.size(0)+1,S.size(1)+1).long()
    for i in range(len(top_id)):
        seleted[i,top_id[i]] = 1
    
    row_nonzero_seleted = torch.nonzero(seleted).numpy().tolist()
    col_nonzero_seleted = torch.nonzero(seleted.T).numpy().tolist()
    temp = torch.zeros(seleted.size(0)).long()
    for line in row_nonzero_seleted:
        seleted[line[0],line[1]] = temp[line[0]]
        temp[line[0]] += 1
    
    quick_col_constraint = [[] for i in range(S.size(1))]
    for line in col_nonzero_seleted:#same as in not quick: j = line[0], i = line[1], k = seleted[line[1],line[0]]
        if line[0] < S.size(1):
            quick_col_constraint[line[0]].append([line[1],seleted[line[1],line[0]].numpy().tolist()])
    print('quick_col_constraint generating time=',time.time()-time_start)
    c = [[] for i in range(S.size(0))]
    for i in range(S.size(0)):
        for j in top_id[i]:
            if j < S.size(1):
                c[i].append(S[i,j].numpy().tolist())
            elif j == S.size(1):
                c[i].append(S_empty_col[i,0].numpy().tolist())
    
    c = c+S_empty_row.numpy().tolist()
    return c, top_id, quick_col_constraint

def f_p_r(label,pred):
    f = f1_score(label,pred)
    p = precision_score(label,pred)
    r = recall_score(label,pred)
    return f,p,r


def IPconstraint_and_solve(args, S, K=1, mean_row_min = 1, mean_col_min = 1, union=True,
                           test_set=None, quick=True, reverse=False, two_stage=False, relaxed=False):
    #row_min, row_min_ix = S.min(1)
    #col_min, col_min_ix = S.min(0)
    
    S_topk_row, top_id_row = S.topk(K, dim=1, largest=False) if K < S.size(1) else S.topk(S.size(1), dim=1, largest=False)
    S_topk_col, top_id_col = S.topk(K, dim=0, largest=False) if K < S.size(1) else S.topk(S.size(0), dim=0, largest=False)
    S_empty_col = mean_row_min*torch.ones(S.size(0)).unsqueeze(1)
    S_empty_row = mean_col_min*torch.ones(S.size(1)).unsqueeze(0)
    if quick and K<1000:
        c, top_id, col_constraint = quick_extract_col_constraint_list(S, S_topk_row, top_id_row, S_empty_row,
                                                               S_topk_col, top_id_col, S_empty_col, union)
    else:
        c, top_id, col_constraint = extract_col_constraint_list(S, S_topk_row, top_id_row, S_empty_row,
                                                               S_topk_col, top_id_col, S_empty_col, union)
    '''
    the integer programming problem:
    '''
    
    IPmodel = Model()
    x = [[IPmodel.add_var(var_type=BINARY) for j in range(len(c[i]))] for i in range(len(c))]
    IPmodel.objective = minimize(xsum(c[i][j]*x[i][j] for i in range(len(x)) for j in range(len(x[i]))))
    for i in range(len(x)-1):
        IPmodel += xsum(x[i][j] for j in range(len(x[i]))) == 1#every entity in g1 only match to less than 1 entity in g2
    
    #add col_constraint:
    for i in range(len(col_constraint)):
        IPmodel += xsum(x[j[0]][j[1]] for j in col_constraint[i]) == 1#every entity in g2 only match to less than 1 entity in g1
    
    IPmodel.optimize()
    
    result_g1 = [[] for i in range(len(top_id))]
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j].x > 0:
                result_g1[i].append(top_id[i][j])
    
    if test_set is None:
        right, wrong = 0, 0
        wrong_case = []
        for pair in data.train_set:
            if reverse:
                pair = [pair[1],pair[0]]
            pred = result_g1[pair[0]][0]
            true = pair[1]
            if pred == true:
                right += 1
            else:
                wrong += 1
                wrong_case.append([pair,result_g1[pair[0]]])
        train_hit1 = right/data.train_set.size(0)
        print('train_hit1:', train_hit1)
        
        right, wrong = 0, 0
        wrong_case = []
        for pair in data.test_set:
            if reverse:
                pair = [pair[1],pair[0]]
            pred = result_g1[pair[0]][0]
            true = pair[1]
            if pred == true:
                right += 1
            else:
                wrong += 1
                wrong_case.append([pair,result_g1[pair[0]]])
        test_hit1 = right/data.test_set.size(0)
        print('test_hit1:', test_hit1)
        
        if relaxed or 'DBP15K' in args.data:
            f,p,r,pred1,pred2 = 0,0,0,[],[]
        else:#dangling testing
            if reverse:
                index_1to2 = torch.cat([data.fixed_test_matchable_idx_2,data.fixed_test_dangling_idx_2]).numpy().tolist()
                label_1to2 = torch.cat([torch.zeros(data.fixed_test_matchable_idx_2.size(0)),torch.ones(data.fixed_test_dangling_idx_2.size(0))]).long().numpy().tolist()
            else:
                index_1to2 = torch.cat([data.fixed_test_matchable_idx_1,data.fixed_test_dangling_idx_1]).numpy().tolist()
                label_1to2 = torch.cat([torch.zeros(data.fixed_test_matchable_idx_1.size(0)),torch.ones(data.fixed_test_dangling_idx_1.size(0))]).long().numpy().tolist()
            pred1, pred2 = [], []
            for i in index_1to2:
                if x[i][-1].x > 0:#i is dangling
                    pred1.append(1)
                else:
                    pred1.append(0)
            f,p,r=f_p_r(label_1to2,pred1)
        if two_stage:
            pred1, pred2 = [], []
            for i in range(len(x)-1):
                if x[i][-1].x > 0:#i is dangling
                    pred1.append(1)
                else:
                    pred1.append(0)
            for j in range(len(x[-1])):
                if x[-1][j].x > 0:#i is dangling
                    pred2.append(1)
                else:
                    pred2.append(0)
        return train_hit1, test_hit1, f, p, r, pred1, pred2
    else:#test_set only
        right, wrong = 0, 0
        wrong_case = []
        for i in range(len(result_g1)-1):
            if i == result_g1[i][0]:
                right += 1
            else:
                wrong += 1
                wrong_case.append([i, result_g1[i]])
        test_hit1 = right/test_set.size(0)
        print('constraint_hit1:', test_hit1)
        return test_hit1

def test_hard(args, model, dangling_model, data, stable=False,file=None,dangling_criterion=None,dangling_optimizer=None):
    x1, x2 = get_emb(model, data)
    print('Train_set testing')
    print('-'*16+'Train_set'+'-'*16, file=file)
    get_hits_hard(x1, x2, data.train_set,file=file)
    
    print('Test_set testing')
    print('-'*16+'Test_set'+'-'*17,file=file)
    get_hits_hard(x1, x2, data.test_set,file=file)
    if stable:
        print('Test_set stable testing')
        get_hits_stable_hard(x1, x2, data.test_set, file=file)
    
    print('done!')
    print(file=file)

def test(args, model, dangling_model, data, stable=False,file=None,dangling_criterion=None,dangling_optimizer=None):
    x1, x2 = get_emb(model, data)
    print('Train_set testing')
    print('-'*16+'Train_set'+'-'*16, file=file)
    get_hits(x1, x2, data.train_set,file=file)
    
    print('Test_set testing')
    print('-'*16+'Test_set'+'-'*17,file=file)
    get_hits(x1, x2, data.test_set,file=file)
    if stable:
        print('Test_set stable testing')
        get_hits_stable(x1, x2, data.test_set, file=file)
    
    print('done!')
    print(file=file)



if __name__ == "__main__":
    args = parse_args()
    reverse = args.reverse
    device = 'cpu'
    data = init_data(args, device).to(device)
    log_path = '/'+args.log_path
    model = torch.load(args.data+'/'+args.lang+log_path+'best.pkl').to(device)
    data.x1 = torch.load(args.data+'/'+args.lang+log_path+'best.pklx1').to(device)
    data.x2 = torch.load(args.data+'/'+args.lang+log_path+'best.pklx2').to(device)
    model.eval()
    x1, x2 = get_emb(model, data)
    if reverse:
        log_path += '_reverse'
        S = torch.cdist(x2, x1, p=1).cpu()
    elif args.csls:
        log_path += '_csls'
        from csls import sim
        S = sim(x1, x2, metric='manhattan', normalize=False, csls_k=10)
        S = -torch.tensor(S)
    else:
        S = torch.cdist(x1, x2, p=1).cpu()
    
    import matplotlib.pyplot as plt
    row_min, row_min_ix = S.min(1)
    col_min, col_min_ix = S.min(0)
    row_n, row_bins, row_patches = plt.hist(row_min.numpy(),100,density = True,color = 'green')
    col_n, col_bins, col_patches = plt.hist(col_min.numpy(),100,density = True,color = 'green')
    
    file = open('./global_result'+log_path+args.lang+'.txt','a+',encoding='utf-8')
    
    
    train_hit1_list = []
    hit1_stable = []
    for mean_row_min, mean_col_min in tqdm(zip(row_bins[:50], col_bins[:50])):
        train_hit1,test_hit1,f,p,r,pred1,pred2 = IPconstraint_and_solve(args, S, 10, mean_row_min, mean_col_min, union=True, reverse=reverse,relaxed=args.relaxed)
        hit1_stable.append([mean_row_min, mean_col_min, train_hit1, test_hit1, f, p, r])
        train_hit1_list.append(train_hit1)
    
    print("union=True, K=10", file=file)
    print(np.array(hit1_stable), file=file)
    
    star_i = np.where(np.array(train_hit1_list)==max(train_hit1_list))[0].min()
    a_star = row_bins[star_i]
    b_star = col_bins[star_i]
    
    
    hit1_stable = []
    train_hit1,test_hit1,f,p,r,pred1,pred2 = IPconstraint_and_solve(args, S, 1, a_star, b_star, union=True, reverse=reverse,relaxed=args.relaxed)
    hit1_stable.append([mean_row_min, mean_col_min, train_hit1, test_hit1, f, p, r])
    print("union=True, K=1", file=file)
    print(np.array(hit1_stable), file=file)
    
    hit1_stable = []
    train_hit1,test_hit1,f,p,r,pred1,pred2 = IPconstraint_and_solve(args, S, 100, a_star, b_star, union=True, reverse=reverse,relaxed=args.relaxed)
    hit1_stable.append([a_star, b_star, train_hit1, test_hit1, f, p, r])
    
    print("union=True, K=100, star_i=", star_i, file=file)
    print(np.array(hit1_stable), file=file)
    
    
    hit1_stable = []
    for mean_row_min, mean_col_min in tqdm(zip(row_bins[-1:], col_bins[-1:])):
        train_hit1,test_hit1,f,p,r,pred1,pred2 = IPconstraint_and_solve(args, S, 100, mean_row_min, mean_col_min, union=True, reverse=reverse,relaxed=args.relaxed)
        hit1_stable.append([mean_row_min, mean_col_min, train_hit1, test_hit1, f, p, r])
    
    print("union=True, K=100, row_bins[-1:], col_bins[-1:]", file=file)
    print(np.array(hit1_stable), file=file)
    
    
    
    if args.csls:
        from csls import sim
        S_train = sim(x1[data.train_set[:,0]], x2[data.train_set[:,1]], metric='manhattan', normalize=False, csls_k=20)
        S_train = -torch.tensor(S_train)
        S_test = sim(x1[data.test_set[:,0]], x2[data.test_set[:,1]], metric='manhattan', normalize=False, csls_k=20)
        S_test = -torch.tensor(S_test)
    else:
        S_train = torch.cdist(x1[data.train_set[:,0]], x2[data.train_set[:,1]], p=1).cpu()
        S_test = torch.cdist(x1[data.test_set[:,0]], x2[data.test_set[:,1]], p=1).cpu()
    
    
    constraint_hit1_stable = []
    for mean_row_min, mean_col_min in tqdm(zip(row_bins[-1:], col_bins[-1:])):
        train_hit1 = IPconstraint_and_solve(args, S_train, 100, mean_row_min, mean_col_min, union=True, test_set = data.train_set, reverse=reverse,relaxed=args.relaxed)
        test_hit1 = IPconstraint_and_solve(args, S_test, 100, mean_row_min, mean_col_min, union=True, test_set = data.test_set, reverse=reverse,relaxed=args.relaxed)
        constraint_hit1_stable.append([mean_row_min, mean_col_min, train_hit1, test_hit1])
    
    print('relaxed_setting_hit1_stable,K=100,union=True', file=file)
    print(np.array(constraint_hit1_stable), file=file)
    
    
    print('-'*16+'relaxed setting'+'-'*17,file=file)
    test(args, model, None, data, stable=True,file=file)
    
    print('-'*16+'practical setting'+'-'*17,file=file)
    test_hard(args, model, None, data, stable=True,file=file)
    
    
    file.close()

    


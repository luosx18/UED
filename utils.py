import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch

def get_neg(x1, x2, j, train_set, k=5):
    print("get_neg:",j)
    neg = []
    for i in range(train_set.size()[0]):
        neg.append(torch.cdist(x1[train_set[i, j]].unsqueeze(0), x2, p=1).topk(k+1, largest=False)[1][:,1:].numpy().tolist()[0])
    neg = torch.tensor(neg).t()
    return neg

def get_train_batch_low_memory(x1, x2, train_set, k=5):
    e1_neg1 = get_neg(x1, x1, 0, train_set, k=5)#torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = get_neg(x1, x2, 0, train_set, k=5)#torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = get_neg(x2, x2, 1, train_set, k=5)#torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = get_neg(x2, x1, 1, train_set, k=5)#torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch

from sklearn.metrics import f1_score, precision_score, recall_score
def dangling_eval(logit, label=None, threshold=0.5, grid=False):
    matchable_logit = logit[:,1]
    pred = torch.where(matchable_logit>threshold, torch.ones_like(matchable_logit),torch.zeros_like(matchable_logit))
    if grid:
        return torch.nonzero(pred)
    f1 = f1_score(label,pred)
    precision = precision_score(label,pred)
    recall = recall_score(label,pred)
    return f1, precision, recall
    

def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10),file=None):
    mean_Hk = 0
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t',end='',file=file)
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        if k == 1:
            mean_Hk += Hk
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='',file=file)
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR,file=file)
    print('Right:\t',end='',file=file)
    for k in Hn_nums:
        pred_topk= S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        if k == 1:
            mean_Hk += Hk
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='',file=file)
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR,file=file)
    return mean_Hk/2, MRR

    
def get_hits_stable(x1, x2, pair,file=None):
    pair_num = pair.size(0)
    S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    #index = S.flatten().argsort(descending=True)
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index//pair_num
    index_e2 = index%pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned/pair_num*100),file=file)


def get_hits_hard(x1, x2, pair, dist='L1', Hn_nums=(1, 10),file=None,wrong=0):
    mean_Hk = 0
    pair_num = pair.size(0)
    S = torch.cdist(x1, x2, p=1)
    print('Left:\t',end='',file=file)
    for k in Hn_nums:
        pred_topk= S[pair[:,0]].topk(k, largest=False)[1]
        Hk1 = (pred_topk == pair[:,1].view(-1, 1)).view(-1, 1).sum().item()/(pair_num+wrong)
        if k == 1:
            mean_Hk += Hk1
        print('Hits@%d: %.2f%%    ' % (k, Hk1*100),end='',file=file)
    rank = torch.where(S[pair[:,0]].sort()[1] == pair[:,1].view(-1, 1))[1].float()
    MRR1 = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR1,file=file)
    print('Right:\t',end='',file=file)
    for k in Hn_nums:
        pred_topk= S.t()[pair[:,1]].topk(k, largest=False)[1]
        Hk2 = (pred_topk == pair[:,0].view(-1, 1)).view(-1, 1).sum().item()/(pair_num+wrong)
        if k == 1:
            mean_Hk += Hk2
        print('Hits@%d: %.2f%%    ' % (k, Hk2*100),end='',file=file)
    rank = torch.where(S.t()[pair[:,1]].sort()[1] == pair[:,0].view(-1, 1))[1].float()
    MRR2 = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR2,file=file)
    return mean_Hk/2, (MRR1+MRR2)/2

    
def get_hits_stable_hard(x1, x2, pair,file=None,wrong=0):
    pair_num = pair.size(0)
    S = torch.cdist(x1, x2, p=1).cpu()
    #index = S.flatten().argsort(descending=True)
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=False)
    index_e1 = index//x2.size(0)
    index_e2 = index%x2.size(0)
    aligned_e1 = torch.zeros(x1.size(0), dtype=torch.bool)
    aligned_e2 = torch.zeros(x2.size(0), dtype=torch.bool)
    true_aligned = 0
    for _ in range(max(x1.size(0),x2.size(0))*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] in pair[:,0] and pair[torch.where(pair[:,0] == index_e1[_])[0][0],1] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    Hits_stable = true_aligned/(pair_num+wrong)*100
    print('Both:\tHits@Stable: %.2f%%    ' % (Hits_stable),file=file)
    return Hits_stable

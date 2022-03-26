import os
import json
import torch
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import sort_edge_index


class DBP15K(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.3, seed=1, 
                 random_init=False, k_init=False, mt_sim_topK='mt_sim_topK3', 
                 mt_pair_file='mt_pair99', mt_pair=False, mt_anchor_weight=False):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.seed = seed
        torch.manual_seed(seed)
        self.random_init = random_init
        self.k_init = k_init
        self.mt_sim_topK = mt_sim_topK
        self.mt_pair_file = mt_pair_file
        self.mt_pair = mt_pair
        self.mt_anchor_weight = mt_anchor_weight
        super(DBP15K, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en', 'es_en']

    @property
    def processed_file_names(self):
        file_names = '%s_%d_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.seed)
        if self.k_init:
            file_names = 'k_init_' + file_names
        if self.mt_pair:
            file_names = 'mt_pair_' + file_names
        if self.mt_anchor_weight:
            file_names = 'mt_anchor_weight_' + file_names
        if self.random_init:
            file_names = 'random_init_' + file_names
        return file_names

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, self.pair[:2]+'_vectorList.json')
        x1, edge_index1, rel1, assoc1 = self.process_graph(g1_path, x1_path, emb_path)
        x2, edge_index2, rel2, assoc2 = self.process_graph(g2_path, x2_path, emb_path, k_init = self.k_init)
        self.assoc1, self.assoc2 = assoc1, assoc2
        print('assoc1',assoc1)
        print('assoc2',assoc2)
        
        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate*pair_set.size(1))]
        test_set = pair_set[:, int(self.rate*pair_set.size(1)):]
        
        '''pseudo pairs: mt_pair'''
        if self.mt_pair:
            mt_pair_path = os.path.join(self.root, self.pair, self.mt_pair_file)
            mt_pair_set = self.process_mt_pair(mt_pair_path, assoc1, assoc2)
        else:
            mt_pair_set = torch.zeros(1)
        
        if self.mt_anchor_weight:
            print('get mt_anchor_weight')
            anchor_list,weight_list,anchor_pair_list = self.load_anchor_weight(assoc1, assoc2)
        else:
            anchor_list,weight_list,anchor_pair_list = torch.zeros(1),torch.zeros(1),torch.zeros(1)

        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, 
                        x2=x2, edge_index2=edge_index2, rel2=rel2, 
                        train_set=train_set.t(), test_set=test_set.t(),
                        mt_pair_set=mt_pair_set.t(),anchor_list=anchor_list,weight_list=weight_list,
                        anchor_pair_list=anchor_pair_list.t())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2+x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2+rel1.max()+1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel,train_set=train_set.t(), test_set=test_set.t(),
                        mt_pair_set=mt_pair_set.t(),anchor_list=anchor_list,weight_list=weight_list,
                        anchor_pair_list=anchor_pair_list.t())
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def load_anchor_weight(self, assoc1=None, assoc2=None):
        anchor_list,weight_list = [],[]
        anchor_pair_list = []
        with open(os.path.join(self.root, self.pair, self.mt_sim_topK), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                anchor,weight = [], []
                for i in range(1,len(line)):
                    if i%2 == 1:
                        anchor.append(int(line[i]))
                        anchor_pair_list.append([int(line[0]),int(line[i])])
                    else:
                        weight.append(float(line[i]))
                anchor_list.append(anchor)
                weight_list.append(weight)
        anchor_list = torch.tensor(anchor_list)
        weight_list = torch.tensor(weight_list)
        anchor_pair_list = torch.tensor(anchor_pair_list)
        #mt_anchor_pair G2 → G1 to G1 → G2
        if assoc1 is not None:
            anchor_pair_list = torch.stack([self.assoc1[anchor_pair_list[:,1]], self.assoc2[anchor_pair_list[:,0]]], dim=0)
        weight_list = weight_list-weight_list.min(1)[0].unsqueeze(1)+0.1*torch.ones(weight_list.size())
        #weight_list = torch.pow(weight_list,2)
        weight_list = weight_list/weight_list.sum(1).unsqueeze(1)
        return anchor_list,weight_list,anchor_pair_list
    
    def process_graph(self, triple_path, ent_path, emb_path, k_init=False):
        g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g.t()
        
        assoc = torch.full((rel.max().item()+1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]
        
        idx = []
        with open(ent_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            if 'DBP' in emb_path:
                embedding_list = torch.tensor(json.load(f))
            else:
                embedding_list = torch.tensor(list(json.load(f).values()))
        if k_init:
            print('init emb with k anchor in graph A')
            anchor_list,weight_list,anchor_pair_list = self.load_anchor_weight()
            print(idx.size()[0],anchor_list.size()[0])
            assert idx.size()[0] == anchor_list.size()[0]
            x = torch.matmul(weight_list.unsqueeze(1),embedding_list[anchor_list]).squeeze(1)
        else:
            x = embedding_list[idx]
        if self.random_init:
            print('random_init')
            x = torch.rand(x.size())#torch.nn.Embedding(num_embeddings=x.size()[0],embedding_dim=x.size()[1]).weight
        
                
                    
                    
        assoc = torch.full((idx.max().item()+1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)
        return x, edge_index, rel, assoc

    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
    def process_mt_pair(self, path, assoc1, assoc2):
        e2, e1 = read_txt_array(path, sep='\t', dtype=torch.long).t()#mt_pair G2 → G1
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)




class MedED(InMemoryDataset):
    def __init__(self, root, pair, KG_num=1, rate=0.3, seed=1, 
                 random_init=False, k_init=False, mt_sim_topK='glove_mt_sim_topK3', 
                 mt_pair_file='glove_mt_pair99', mt_pair=False, mt_anchor_weight=False):
        self.pair = pair
        self.KG_num = KG_num
        self.rate = rate
        self.seed = seed
        torch.manual_seed(seed)
        self.random_init = random_init
        self.k_init = k_init
        self.mt_sim_topK = mt_sim_topK
        self.mt_pair_file = mt_pair_file
        self.mt_pair = mt_pair
        self.mt_anchor_weight = mt_anchor_weight
        super(MedED, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return ['zh_en', 'fr_en', 'ja_en', 'es_en']

    @property
    def processed_file_names(self):
        file_names = '%s_%d_%.1f_%d.pt' % (self.pair, self.KG_num, self.rate, self.seed)
        if self.k_init:
            file_names = 'k_init_' + file_names
        if self.mt_pair:
            file_names = 'mt_pair_' + file_names
        if self.mt_anchor_weight:
            file_names = 'mt_anchor_weight_' + file_names
        if self.random_init:
            file_names = 'random_init_' + file_names
        
        return file_names

    def process(self):
        x1_path = os.path.join(self.root, self.pair, 'ent_ids_1')
        x2_path = os.path.join(self.root, self.pair, 'ent_ids_2')
        g1_path = os.path.join(self.root, self.pair, 'triples_1')
        g2_path = os.path.join(self.root, self.pair, 'triples_2')
        emb_path = os.path.join(self.root, self.pair, self.pair[:2]+'_vectorList.json')
        x1, edge_index1, rel1, assoc1 = self.process_graph(g1_path, x1_path, emb_path)
        x2, edge_index2, rel2, assoc2 = self.process_graph(g2_path, x2_path, emb_path, k_init = self.k_init)
        self.assoc1, self.assoc2 = assoc1, assoc2
        print('assoc1',assoc1)
        print('assoc2',assoc2)
        
        pair_path = os.path.join(self.root, self.pair, 'ref_ent_ids')
        pair_set = self.process_pair(pair_path, assoc1, assoc2)
        pair_set = pair_set[:, torch.randperm(pair_set.size(1))]
        train_set = pair_set[:, :int(self.rate*pair_set.size(1))]
        test_set = pair_set[:, int(self.rate*pair_set.size(1)):]
        
        
        '''dangling'''
        if 'DBP' not in self.root:
            dangling_x1_path = os.path.join(self.root, self.pair, 'dangling_ids_1')
            dangling_x2_path = os.path.join(self.root, self.pair, 'dangling_ids_2')
            
            dangling_idx_1 = self.process_dangling_entity(dangling_x1_path, assoc1)
            dangling_idx_2 = self.process_dangling_entity(dangling_x2_path, assoc2)
            #random permutation
            dangling_idx_1 = dangling_idx_1[torch.randperm(dangling_idx_1.size(0))]
            dangling_idx_2 = dangling_idx_2[torch.randperm(dangling_idx_2.size(0))]


            #split train test
            train_dangling_idx_1 = dangling_idx_1[:int(self.rate*dangling_idx_1.size(0))]
            test_dangling_idx_1 = dangling_idx_1[int(self.rate*dangling_idx_1.size(0)):]
            train_dangling_idx_2 = dangling_idx_2[:int(self.rate*dangling_idx_2.size(0))]
            test_dangling_idx_2 = dangling_idx_2[int(self.rate*dangling_idx_2.size(0)):]
            

            train_matchable_idx_1 = train_set[0,:]
            test_matchable_idx_1 = test_set[0,:]
            train_matchable_idx_2 = train_set[1,:]
            test_matchable_idx_2 = test_set[1,:]
        
        
        '''pseudo_pair'''
        if self.mt_pair:
            mt_pair_path = os.path.join(self.root, self.pair, self.mt_pair_file)
            mt_pair_set = self.process_mt_pair(mt_pair_path, assoc1, assoc2)
        else:
            mt_pair_set = torch.zeros(1)
        
        if self.mt_anchor_weight:
            print('get mt_anchor_weight')
            anchor_list,weight_list,anchor_pair_list = self.load_anchor_weight(assoc1, assoc2)
        else:
            anchor_list,weight_list,anchor_pair_list = torch.zeros(1),torch.zeros(1),torch.zeros(1)

        if self.KG_num == 1:
            data = Data(x1=x1, edge_index1=edge_index1, rel1=rel1, 
                        x2=x2, edge_index2=edge_index2, rel2=rel2, 
                        train_set=train_set.t(), test_set=test_set.t(),
                        mt_pair_set=mt_pair_set.t(),anchor_list=anchor_list,weight_list=weight_list,
                        anchor_pair_list=anchor_pair_list.t(),
                        fixed_train_dangling_idx_1 = train_dangling_idx_1.clone(),fixed_test_dangling_idx_1 = test_dangling_idx_1.clone(),
                        fixed_train_dangling_idx_2 = train_dangling_idx_2.clone(),fixed_test_dangling_idx_2 = test_dangling_idx_2.clone(),
                        train_dangling_idx_1 = train_dangling_idx_1,test_dangling_idx_1 = test_dangling_idx_1,
                        train_dangling_idx_2 = train_dangling_idx_2,test_dangling_idx_2 = test_dangling_idx_2,
                        train_matchable_idx_1 = train_matchable_idx_1,test_matchable_idx_1 = test_matchable_idx_1,
                        train_matchable_idx_2 = train_matchable_idx_2,test_matchable_idx_2 = test_matchable_idx_2,
                        fixed_train_matchable_idx_1 = train_matchable_idx_1.clone(),fixed_test_matchable_idx_1 = test_matchable_idx_1.clone(),
                        fixed_train_matchable_idx_2 = train_matchable_idx_2.clone(),fixed_test_matchable_idx_2 = test_matchable_idx_2.clone())
        else:
            x = torch.cat([x1, x2], dim=0)
            edge_index = torch.cat([edge_index1, edge_index2+x1.size(0)], dim=1)
            rel = torch.cat([rel1, rel2+rel1.max()+1], dim=0)
            data = Data(x=x, edge_index=edge_index, rel=rel,train_set=train_set.t(), test_set=test_set.t(),
                        mt_pair_set=mt_pair_set.t(),anchor_list=anchor_list,weight_list=weight_list,
                        anchor_pair_list=anchor_pair_list.t(),
                        fixed_train_dangling_idx_1 = train_dangling_idx_1.clone(),fixed_test_dangling_idx_1 = test_dangling_idx_1.clone(),
                        fixed_train_dangling_idx_2 = train_dangling_idx_2.clone(),fixed_test_dangling_idx_2 = test_dangling_idx_2.clone(),
                        train_dangling_idx_1 = train_dangling_idx_1,test_dangling_idx_1 = test_dangling_idx_1,
                        train_dangling_idx_2 = train_dangling_idx_2,test_dangling_idx_2 = test_dangling_idx_2,
                        train_matchable_idx_1 = train_matchable_idx_1,test_matchable_idx_1 = test_matchable_idx_1,
                        train_matchable_idx_2 = train_matchable_idx_2,test_matchable_idx_2 = test_matchable_idx_2,
                        fixed_train_matchable_idx_1 = train_matchable_idx_1.clone(),fixed_test_matchable_idx_1 = test_matchable_idx_1.clone(),
                        fixed_train_matchable_idx_2 = train_matchable_idx_2.clone(),fixed_test_matchable_idx_2 = test_matchable_idx_2.clone())
        torch.save(self.collate([data]), self.processed_paths[0])
    
    def load_anchor_weight(self, assoc1=None, assoc2=None):
        anchor_list,weight_list = [],[]
        anchor_pair_list = []
        with open(os.path.join(self.root, self.pair, self.mt_sim_topK), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                anchor,weight = [], []
                for i in range(1,len(line)):
                    if i%2 == 1:
                        anchor.append(int(line[i]))
                        anchor_pair_list.append([int(line[0]),int(line[i])])
                    else:
                        weight.append(float(line[i]))
                anchor_list.append(anchor)
                weight_list.append(weight)
        anchor_list = torch.tensor(anchor_list)
        weight_list = torch.tensor(weight_list)
        anchor_pair_list = torch.tensor(anchor_pair_list)
        #mt_anchor_pair G2 → G1 to G1 → G2
        if assoc1 is not None:
            anchor_pair_list = torch.stack([self.assoc1[anchor_pair_list[:,1]], self.assoc2[anchor_pair_list[:,0]]], dim=0)
        weight_list = weight_list-weight_list.min(1)[0].unsqueeze(1)+0.1*torch.ones(weight_list.size())
        #weight_list = torch.pow(weight_list,2)
        weight_list = weight_list/weight_list.sum(1).unsqueeze(1)
        return anchor_list,weight_list,anchor_pair_list
    
    def process_graph(self, triple_path, ent_path, emb_path, k_init=False):
        g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
        subj, rel, obj = g.t()
        
        assoc = torch.full((rel.max().item()+1,), -1, dtype=torch.long)
        assoc[rel.unique()] = torch.arange(rel.unique().size(0))
        rel = assoc[rel]
        
        idx = []
        with open(ent_path, 'r') as f:
            for line in f:
                info = line.strip().split('\t')
                idx.append(int(info[0]))
        idx = torch.tensor(idx)
        with open(emb_path, 'r', encoding='utf-8') as f:
            if 'DBK' in emb_path:
                embedding_list = torch.tensor(json.load(f))
            else:
                embedding_list = torch.tensor(list(json.load(f).values()))
        if k_init:
            print('init emb with k anchor in graph A')
            anchor_list,weight_list,anchor_pair_list = self.load_anchor_weight()
            print(idx.size()[0],anchor_list.size()[0])
            assert idx.size()[0] == anchor_list.size()[0]
            x = torch.matmul(weight_list.unsqueeze(1),embedding_list[anchor_list]).squeeze(1)
        else:
            x = embedding_list[idx]
        if self.random_init:
            print('random_init')
            x = torch.rand(x.size())#torch.nn.Embedding(num_embeddings=x.size()[0],embedding_dim=x.size()[1]).weight
        
                
        
                    
        assoc = torch.full((idx.max().item()+1, ), -1, dtype=torch.long)
        assoc[idx] = torch.arange(idx.size(0))
        subj, obj = assoc[subj], assoc[obj]
        edge_index = torch.stack([subj, obj], dim=0)
        edge_index, rel = sort_edge_index(edge_index, rel)
        return x, edge_index, rel, assoc

    def process_pair(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
    def process_mt_pair(self, path, assoc1, assoc2):
        e2, e1 = read_txt_array(path, sep='\t', dtype=torch.long).t()#mt_pair G2 → G1
        return torch.stack([assoc1[e1], assoc2[e2]], dim=0)
    def process_dangling_entity(self,path, assoc):
        e = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return assoc[e]
    def process_matchable_entity(self, path, assoc1, assoc2):
        e1, e2 = read_txt_array(path, sep='\t', dtype=torch.long).t()
        return assoc1[e1], assoc2[e2]

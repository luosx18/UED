# -*- coding: utf-8 -*-


import numpy as np
import torch



'''load mt_es2en and en'''
fr_id, en_id, mt_fr2en, en = [], [], [], []
with open('./MedED/es_en/ent_ids_2_tran_1','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        fr_id.append(line[0])
        mt_fr2en.append(line[1])

with open('./MedED/es_en/ent_ids_1','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        en_id.append(line[0])
        en.append(line[1])

def get_glove_emb(fr_id, ix2vec):
    fr_emb = []
    for i in fr_id:
        vec = ix2vec[i]
        norm = np.linalg.norm(vec)
        if norm > 0:
            fr_emb.append(vec/norm)
        else:
            fr_emb.append(vec)
    return fr_emb

import json
ix2vec = json.load(open('./MedED/es_en/es_vectorList.json'))
fr_emb = get_glove_emb(fr_id, ix2vec)
en_emb = get_glove_emb(en_id, ix2vec)

def sim(a,b):
    return np.sum(a*b)

sim_mat = np.matmul(fr_emb, np.transpose(en_emb))

mt_matched_pair = []
threshold = 0.99
s = np.where(sim_mat>threshold)
print(s[0].shape,len(set(s[0])))
fr_en = {}
for i in range(s[0].shape[0]):
    if s[0][i] not in fr_en:
        fr_en[s[0][i]] = []
    fr_en[s[0][i]].append(s[1][i])

for i in fr_en:
    if len(fr_en[i]) > 1:
        print(fr_id[i],fr_en[i])

with open('./MedED/es_en/glove_mt_pair99','w',encoding='utf-8') as f:
    for i in range(s[0].shape[0]):
        temp = f.write(str(fr_id[s[0][i]])+'\t'+str(en_id[s[1][i]])+'\n')


K = 3
sim_mat = torch.tensor(sim_mat)
_, indices = sim_mat.topk(K)
indices = indices.numpy().tolist()
sim_mat = sim_mat.numpy()
with open('./MedED/es_en/glove_mt_sim_topK3','w',encoding='utf-8') as f:
    for i in range(len(indices)):
        line = str(fr_id[i])+'\t'
        for j in indices[i][:-1]:
            line += str(en_id[j])
            line += '\t'
            line += str(sim_mat[i,j])
            line += '\t'
        j = indices[i][-1]
        line += str(en_id[j])
        line += '\t'
        line += str(sim_mat[i,j])
        temp = f.write(line+'\n')






# =============================================================================
# device = "cuda:0"
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("../coder/upload_model/UMLSBert_ENG")
# model = AutoModel.from_pretrained("../coder/upload_model/UMLSBert_ENG").to(device)
# 
# 
# model.eval()
# 
# normalize = True
# summary_method = "MEAN"
# 
# def get_coder_emb(phrase_list,device = "cpu",normalize = True, summary_method = "MEAN"):
#     input_ids = []
#     for phrase in phrase_list:
#         input_ids.append(tokenizer.encode_plus(
#             phrase, max_length=32, add_special_tokens=True,
#             truncation=True, pad_to_max_length=True)['input_ids'])
#     now_count = 0
#     batch_size = 32
#     count = len(input_ids)
#     with torch.no_grad():
#         while now_count < count:
#             input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
#                 now_count + batch_size, count)]).to(device)
#             if summary_method == "CLS":
#                 embed = model(input_gpu_0)[1]
#             if summary_method == "MEAN":
#                 embed = torch.mean(model(input_gpu_0)[0], dim=1)
#             if normalize:
#                 embed_norm = torch.norm(
#                     embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
#                 embed = embed / embed_norm
#             embed_np = embed.cpu().detach().numpy()
#             if now_count == 0:
#                 output = embed_np
#             else:
#                 output = np.concatenate((output, embed_np), axis=0)
#             now_count = min(now_count + batch_size, count)
#     return output
# =============================================================================


'''use coder as LM'''
#fr_emb = get_coder_emb(mt_fr2en,device=device)
#en_emb = get_coder_emb(en,device=device)







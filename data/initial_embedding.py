# -*- coding: utf-8 -*-


from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

'''load glove'''
glove_input_file = 'glove.840B.300d.txt'
word2vec_output_file = 'glove.840B.300d.word2vec.txt'

glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
cat_vec = glove_model['cat']



def read_cui2ix(file):
    cui2ix = {}
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            cui2ix[line[1]] = int(line[0])
    return cui2ix

ENG_cui2ix = read_cui2ix('./MedED/es_en/ENG_cui2ix')
SPA_cui2ix = read_cui2ix('./MedED/es_en/SPA_cui2ix')
ix2cui={}
for cui in ENG_cui2ix:
    ix2cui[ENG_cui2ix[cui]]=cui

for cui in SPA_cui2ix:
    ix2cui[SPA_cui2ix[cui]]=cui


import numpy as np
def term_vec(term,glove_model):
    vec = [glove_model[w] for w in term if w in glove_model]
    if len(vec) > 0:
        return np.mean(vec,axis=0)
    else:
        return np.zeros(cat_vec.shape,dtype='float32')

def read_ix2term(file):
    ix2term = {}
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            ix2term[int(line[0])] = line[1]
    return ix2term

ix2term = {**read_ix2term('./MedED/es_en/ent_ids_1'), **read_ix2term('./MedED/es_en/ent_ids_2_tran_1')}
ix2vec = {}
for i in ix2term:
    term = ix2term[i]
    term = term.split(' ')
    ix2vec[int(i)] = term_vec(term,glove_model).tolist()

import json
with open('./MedED/es_en/es_vectorList.json', 'w') as f:
    json.dump(ix2vec, f)

ix2vec = json.load(open('./MedED/es_en/es_vectorList.json'))
















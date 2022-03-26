# -*- coding: utf-8 -*-

import sub_umls
from tqdm import tqdm
import sys
seleted_num_of_cui = 20000
if len(sys.argv) > 1:
    seleted_num_of_cui = int(sys.argv[1])

def read_sub_cui(file):
    cui_set = []
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            cui_set.append(line)
        cui_set = set(cui_set)
    print(file,len(cui_set))
    return cui_set

def read_seed(file):
    seed = []
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            seed.append(line)
    print('seed size = ',len(seed))
    return seed

umls = sub_umls.sub_UMLS("./",lang_range=None)
ENG_cui = read_sub_cui('./MedED/'+'sub_ENG_cui_'+str(seleted_num_of_cui)+'.txt')
SPA_cui = read_sub_cui('./MedED/'+'sub_SPA_cui_'+str(seleted_num_of_cui)+'.txt')
seed = ENG_cui & SPA_cui
with open('./MedED/es_en/seed_cui_'+str(seleted_num_of_cui),'w',encoding='utf-8') as f:
    for line in seed:
        f.write(line+'\n')
seed = read_seed('./MedED/es_en/seed_cui_'+str(seleted_num_of_cui))


def cui_ix(ENG_cui,base = 0):
    cui2ix = {}
    for cui in ENG_cui:
        cui2ix[cui] = len(cui2ix)+base
    return cui2ix

ENG_cui2ix = cui_ix(ENG_cui)
SPA_cui2ix = cui_ix(SPA_cui,len(ENG_cui2ix))
with open('./MedED/es_en/ENG_cui2ix','w',encoding='utf-8') as f:
    for cui in ENG_cui2ix:
        temp = f.write(str(ENG_cui2ix[cui])+'\t'+cui+'\n')

with open('./MedED/es_en/SPA_cui2ix','w',encoding='utf-8') as f:
    for cui in SPA_cui2ix:
        temp = f.write(str(SPA_cui2ix[cui])+'\t'+cui+'\n')
def read_cui2ix(file):
    cui2ix = {}
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            cui2ix[line[1]] = int(line[0])
    return cui2ix
ENG_cui2ix = read_cui2ix('./MedED/es_en/ENG_cui2ix')
SPA_cui2ix = read_cui2ix('./MedED/es_en/SPA_cui2ix')


'''ent_ids_1=ENG,ent_ids_2=SPA'''
def cui_select_term(cui,lang,umls=umls):
    if cui not in umls.cui2str:
        return 0
    term_list = umls.cui2str[cui]
    str_in_lang = []
    for line in term_list:
        line = line.split('|')
        if line[1] == lang:
            str_in_lang.append(line)
    if len(str_in_lang) == 1:
        return str_in_lang[0][0]
    elif len(str_in_lang) == 0:
        return 'None'
    candidate=[]
    for line in str_in_lang:
        if line[2].lower() == 'p':
            candidate.append(line)
    if len(candidate) == 1:
        return candidate[0][0]
    else:
        #print(cui,'len(candidate)=',len(candidate))
        if len(candidate) > 1:
            return candidate[0][0]
        else:
            return str_in_lang[0][0]

def write_ent_ids(f,ENG_cui2ix,lang,umls):
    for cui in tqdm(ENG_cui2ix):
        term = cui_select_term(cui,lang,umls)
        f.write(str(ENG_cui2ix[cui])+'\t'+term+'\n')

with open('./MedED/es_en/ent_ids_1','w',encoding='utf-8') as f:
    write_ent_ids(f,ENG_cui2ix,'ENG',umls)

with open('./MedED/es_en/ent_ids_2','w',encoding='utf-8') as f:
    write_ent_ids(f,SPA_cui2ix,'SPA',umls)

'''ori_ref_ent_ids: entity alignment for testing, list of pairs like (e_s \t e_t)'''
'''ori_sup_ent_ids: seed entity alignment (training data)'''
seed = read_seed('./MedED/es_en/seed_cui_'+str(seleted_num_of_cui))
import random
random.seed(0);ref = random.sample(seed,int(len(seed)*0.3))
sup = list(set(seed)-set(ref))
with open('./MedED/es_en/ori_ref_ent_ids','w',encoding='utf-8') as f:
    for line in ref:
        f.write(str(ENG_cui2ix[line])+'\t'+str(SPA_cui2ix[line])+'\n')

with open('./MedED/es_en/ori_sup_ent_ids','w',encoding='utf-8') as f:
    for line in sup:
        f.write(str(ENG_cui2ix[line])+'\t'+str(SPA_cui2ix[line])+'\n')

'''rel_ids_1: relation ids in the source KG;'''
'''rel_ids_2: relation ids in the target KG'''
'''rel=tri[3]  re=tri[2]'''
ENG_umls = sub_umls.sub_UMLS("./",lang_range=None,seleted_cui=ENG_cui2ix)
SPA_umls = sub_umls.sub_UMLS("./",lang_range=None,seleted_cui=SPA_cui2ix)
ENG_rel2ix = {};ENG_rel2count = {}
for tri in tqdm(ENG_umls.rel):
    tri = tri.split('\t')
    rel = tri[3];re_ = tri[2]
    if len(rel) < 2:
        continue
    if rel not in ENG_rel2ix:
        ENG_rel2ix[rel] = len(ENG_rel2ix)
    if rel not in ENG_rel2count:
        ENG_rel2count[rel] = 0
    ENG_rel2count[rel] += 1

SPA_rel2ix = {};SPA_rel2count = {}
for tri in tqdm(SPA_umls.rel):
    tri = tri.split('\t')
    rel = tri[3];re_ = tri[2]
    if len(rel) < 2:
        #print(tri)
        continue
    if rel not in SPA_rel2ix:
        SPA_rel2ix[rel] = len(SPA_rel2ix)
    if rel not in SPA_rel2count:
        SPA_rel2count[rel] = 0
    SPA_rel2count[rel] += 1

with open('./MedED/es_en/rel_ids_1','w',encoding='utf-8') as f:
    for re in ENG_rel2ix:
        temp = f.write(str(ENG_rel2ix[re])+'\t'+re+'\n')

with open('./MedED/es_en/rel_ids_2','w',encoding='utf-8') as f:
    for re in SPA_rel2ix:
        temp = f.write(str(SPA_rel2ix[re])+'\t'+re+'\n')

ENG_rel2ix = read_cui2ix('./MedED/es_en/rel_ids_1')
SPA_rel2ix = read_cui2ix('./MedED/es_en/rel_ids_2')

'''triples_1: relation triples in the source KG'''
'''triples_2: relation triples in the target KG'''
with open('./MedED/es_en/triples_1','w',encoding='utf-8') as f:
    for tri in tqdm(ENG_umls.rel):
        tri = tri.split('\t')
        rel = tri[3];re_ = tri[2]
        if len(rel) < 2:
            continue
        temp = f.write(str(ENG_cui2ix[tri[0]])+'\t'+str(ENG_rel2ix[rel])+'\t'+str(ENG_cui2ix[tri[1]])+'\n')

with open('./MedED/es_en/triples_2','w',encoding='utf-8') as f:
    for tri in tqdm(SPA_umls.rel):
        tri = tri.split('\t')
        rel = tri[3];re_ = tri[2]
        if len(rel) < 2:
            continue
        temp = f.write(str(SPA_cui2ix[tri[0]])+'\t'+str(SPA_rel2ix[rel])+'\t'+str(SPA_cui2ix[tri[1]])+'\n')


SPAix_translate_ENG = {}
with open('./MedED/es_en/ent_2_to_1','r',encoding='utf-8') as f1:
    with open('./MedED/es_en/ent_ids_2','r',encoding='utf-8') as f2:
        text1 = f1.readlines()
        text2 = f2.readlines()
        for i in range(len(text1)):
            ix = int(text2[i].split('\t')[0])
            SPAix_translate_ENG[ix]=text1[i].strip('\n')

SPA_cui2ix_20000 = read_cui2ix('./MedED/es_en/SPA_cui2ix')
SPA_ix2cui_20000 = {}
for cui in SPA_cui2ix_20000:
    SPA_ix2cui_20000[SPA_cui2ix_20000[cui]] = cui

SPA_ix2cui = {}
for cui in SPA_cui2ix:
    SPA_ix2cui[SPA_cui2ix[cui]] = cui

    

with open('./MedED/es_en/ent_2_to_1','r',encoding='utf-8') as f1:
    with open('./MedED/es_en/ent_ids_2','r',encoding='utf-8') as f2:
        with open('./MedED/es_en/ent_ids_2_tran_1','w',encoding='utf-8') as f_out:
            text1 = f1.readlines()
            text2 = f2.readlines()
            for i in range(len(text1)):
                f_out.write(text2[i].split('\t')[0]+'\t'+text1[i])
            


with open('./MedED/es_en/ori_sup_ent_ids','r',encoding='utf-8') as f1:
    with open('./MedED/es_en/ori_ref_ent_ids','r',encoding='utf-8') as f2:
        with open('./MedED/es_en/ref_ent_ids','w',encoding='utf-8') as f:
            for line in f1:
                if len(line) > 1:
                    f.write(line)
            for line in f2:
                f.write(line)
            



'''extract all in-graph matchable entities and dangling entities'''
#matchable entities are seeds
dangling_ENG = list(set(ENG_cui2ix)-set(seed))
dangling_SPA = list(set(SPA_cui2ix)-set(seed))
with open('./MedED/es_en/dangling_ids_1','w',encoding='utf-8') as f:
    for cui in dangling_ENG:
        f.write(str(ENG_cui2ix[cui])+'\n')

with open('./MedED/es_en/dangling_ids_2','w',encoding='utf-8') as f:
    for cui in dangling_SPA:
        f.write(str(SPA_cui2ix[cui])+'\n')

with open('./MedED/es_en/matchable_ids','w',encoding='utf-8') as f:
    for cui in seed:
        f.write(str(ENG_cui2ix[cui])+'\t'+str(SPA_cui2ix[cui])+'\n')










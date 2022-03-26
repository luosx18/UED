import os
from tqdm import tqdm
import re
from random import shuffle
import random
import sys

def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return


class sub_UMLS(object):
    def __init__(self, umls_path, source_range=None, lang_range=['ENG'], only_load_dict=False, seleted_cui = None):
        self.seleted_cui = seleted_cui
        self.umls_path = umls_path
        self.source_range = source_range
        self.lang_range = lang_range
        self.detect_type()
        self.load()
        
        if not only_load_dict:
            self.load_rel()
            self.load_sty()

    def detect_type(self):
        if os.path.exists(os.path.join(self.umls_path, "MRCONSO.RRF")):
            self.type = "RRF"
        else:
            self.type = "txt"

    def load(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRCONSO." + self.type))
        self.lui_set = set()
        self.cui2str = {}
        self.str2cui = {}
        self.code2cui = {}
        #self.lui_status = {}
        read_count = 0
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            lang = l[1]
            # lui_status = l[2].lower() # p -> preferred
            lui = l[3]
            source = l[11]
            code = l[13]
            string = l[14]
            
            TS = l[2] #Term Status
            STT = l[4] #String Type

            if (self.seleted_cui is None or cui in self.seleted_cui) and (self.source_range is None or source in self.source_range) and (self.lang_range is None or lang in self.lang_range):
                if not lui in self.lui_set:
                    read_count += 1
                    self.str2cui[string] = cui
                    self.str2cui[string.lower()] = cui
                    clean_string = self.clean(string)
                    self.str2cui[clean_string] = cui

                    if not cui in self.cui2str:
                        self.cui2str[cui] = set()
                    self.cui2str[cui].update([clean_string+'|'+lang+'|'+TS+'|'+STT])
                    self.code2cui[code] = cui
                    self.lui_set.update([lui])

            # For debug
            # if read_count > 1000:
            #     break

        self.cui = list(self.cui2str.keys())
        shuffle(self.cui)
        self.cui_count = len(self.cui)

        print("cui count:", self.cui_count)
        print("str2cui count:", len(self.str2cui))
        print("MRCONSO count:", read_count)

    def cui_lang(self, code, lang):
        for line in self.cui2str[code]:
            line = line.split('|')
            if line[1] == lang:
                return True
        return False
    
    def load_rel(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRREL." + self.type))
        self.rel = set()
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui0 = l[0]
            re = l[3]
            cui1 = l[4]
            rel = l[7]
            if cui0 in self.cui2str and cui1 in self.cui2str and \
                (self.seleted_cui is None or (cui0 in self.seleted_cui and cui1 in self.seleted_cui)):
                str_rel = "\t".join([cui0, cui1, re, rel])
                if not str_rel in self.rel and cui0 != cui1:
                    self.rel.update([str_rel])

            # For debug
            # if len(self.rel) > 1000:
            #     break
        self.rel = list(self.rel)

        print("rel count:", len(self.rel))

    def load_sty(self):
        reader = byLineReader(os.path.join(self.umls_path, "MRSTY." + self.type))
        self.cui2sty = {}
        for line in tqdm(reader, ascii=True):
            if self.type == "txt":
                l = [t.replace("\"", "") for t in line.split(",")]
            else:
                l = line.strip().split("|")
            cui = l[0]
            sty = l[3]
            if cui in self.cui2str and (self.seleted_cui is None or cui in self.seleted_cui):
                self.cui2sty[cui] = sty

        print("sty count:", len(self.cui2sty))

    def clean(self, term, lower=True, clean_NOS=True, clean_bracket=True, clean_dash=True):
        term = " " + term + " "
        if lower:
            term = term.lower()
        if clean_NOS:
            term = term.replace(" NOS ", " ").replace(" nos ", " ")
        if clean_bracket:
            term = re.sub(u"\\(.*?\\)", "", term)
        if clean_dash:
            term = term.replace("-", " ")
        term = " ".join([w for w in term.split() if w])
        return term

    def search_by_code(self, code):
        if code in self.cui2str:
            return list(self.cui2str[code])
        if code in self.code2cui:
            return list(self.cui2str[self.code2cui[code]])
        return None

    def search_by_string_list(self, string_list):
        for string in string_list:
            if string in self.str2cui:
                find_string = self.cui2str[self.str2cui[string]]
                return [string for string in find_string if not string in string_list]
            if string.lower() in self.str2cui:
                find_string = self.cui2str[self.str2cui[string.lower()]]
                return [string for string in find_string if not string in string_list]
        return None

    def search(self, code=None, string_list=None, max_number=-1):
        result_by_code = self.search_by_code(code)
        if result_by_code is not None:
            if max_number > 0:
                return result_by_code[0:min(len(result_by_code), max_number)]
            return result_by_code
        return None
        result_by_string = self.search_by_string_list(string_list)
        if result_by_string is not None:
            if max_number > 0:
                return result_by_string[0:min(len(result_by_string), max_number)]
            return result_by_string
        return None

def merge_dicts(dict_args):
    result = {}
    for item in dict_args:
        result.update(item)
    return result

random.choices(range(3),k=3)  
    
    
    


if __name__ == "__main__":
    if len(sys.argv) > 2:
        seleted_num_of_cui = int(sys.argv[1])
        print('seleted_num_of_cui=',seleted_num_of_cui,type(seleted_num_of_cui))
        for seleted_lang in sys.argv[2:]:
            print('processing ', seleted_lang)
            umls = sub_UMLS("./",lang_range=[seleted_lang])
            print('done ', seleted_lang)
            lang_str_count = {}
            for code in tqdm(umls.cui2str):
                for line in umls.cui2str[code]:
                    line = line.split('|')
                    if line[1] not in lang_str_count:
                        lang_str_count[line[1]] = 0
                    lang_str_count[line[1]] += 1
        
            lang_cui = {}
            cui_lang = {}
            lang_cui_count = {}
            for code in tqdm(umls.cui2str):
                if code not in cui_lang:
                    cui_lang[code] = set()
                for line in umls.cui2str[code]:
                    line = line.split('|')
                    cui_lang[code].update([line[1]])
                    if line[1] not in lang_cui:
                        lang_cui[line[1]] = set()
                    lang_cui[line[1]].update([code])
            for lang in lang_cui:
                lang_cui_count[lang] = len(lang_cui[lang])
            lang_sty = {}
            for code in tqdm(umls.cui2str):
                for line in umls.cui2str[code]:
                    line = line.split('|')
                    if line[1] not in lang_sty:
                        lang_sty[line[1]] = {}
                    sty = umls.cui2sty[code]
                    if sty not in lang_sty[line[1]]:
                        lang_sty[line[1]][sty] = set()
                    lang_sty[line[1]][sty].update([code])
            lang_sty_count = {}
            lang_sty_cui_count = {}
            for lang in lang_sty:
                lang_sty_count[lang] = len(lang_sty[lang])
                if lang not in lang_sty_cui_count:
                    lang_sty_cui_count[lang] = {}
                for sty in lang_sty[lang]:
                    if sty not in lang_sty_cui_count[lang]:
                        lang_sty_cui_count[lang][sty] = {}
                    lang_sty_cui_count[lang][sty] = len(lang_sty[lang][sty])
            
            re_count,rel_count = {},{}
            for tri in tqdm(umls.rel):
                tri = tri.split('\t')
                if tri[2] not in re_count:
                    re_count[tri[2]] = 0
                if tri[3] not in rel_count:
                    rel_count[tri[3]] = 0
                re_count[tri[2]] += 1
                rel_count[tri[3]] += 1
            
            cui2tri_count = {}
            for tri in tqdm(umls.rel):
                tri = tri.split('\t')
                rel = tri[3]
                if len(rel) == 0: 
                    continue
                if tri[0] not in cui2tri_count:
                    cui2tri_count[tri[0]] = 0
                if tri[1] not in cui2tri_count:
                    cui2tri_count[tri[1]] = 0
                cui2tri_count[tri[0]] += 1
                cui2tri_count[tri[1]] += 1
            
            tri_count2distribution = {}
            for cui in cui2tri_count:
                count = cui2tri_count[cui]
                if count not in tri_count2distribution:
                    tri_count2distribution[count] = 0
                tri_count2distribution[count] += 1
            
            cui2tri_count_sorted = sorted(cui2tri_count.items(), key=lambda d:d[1], reverse=True)
            if len(cui2tri_count_sorted) > seleted_num_of_cui:
                print(cui2tri_count_sorted[seleted_num_of_cui])
            else:
                print('len(cui2tri_count)=',len(cui2tri_count))
            
            #sub_graph
            seleted_cui = {}
            for cui in cui2tri_count:
                if cui2tri_count[cui] > cui2tri_count_sorted[seleted_num_of_cui][1]:
                    seleted_cui[cui] = cui2tri_count[cui]
            print(len(seleted_cui))
            sub_graph = sub_UMLS("./",lang_range=[seleted_lang],seleted_cui=seleted_cui)
            
            
            re_count,rel_count = {},{}
            for tri in tqdm(sub_graph.rel):
                tri = tri.split('\t')
                if tri[2] not in re_count:
                    re_count[tri[2]] = 0
                if tri[3] not in rel_count:
                    rel_count[tri[3]] = 0
                re_count[tri[2]] += 1
                rel_count[tri[3]] += 1
            
            cui2tri_count = {}
            for tri in tqdm(sub_graph.rel):
                tri = tri.split('\t')
                rel = tri[3]
                if len(rel) == 0: 
                    continue
                if tri[0] not in cui2tri_count:
                    cui2tri_count[tri[0]] = 0
                if tri[1] not in cui2tri_count:
                    cui2tri_count[tri[1]] = 0
                cui2tri_count[tri[0]] += 1
                cui2tri_count[tri[1]] += 1
            
            tri_count2distribution = {}
            for cui in cui2tri_count:
                count = cui2tri_count[cui]
                if count not in tri_count2distribution:
                    tri_count2distribution[count] = 0
                tri_count2distribution[count] += 1
            
            with open('./sub_'+seleted_lang+'_cui_'+str(seleted_num_of_cui)+'.txt','w',encoding='utf-8') as f:
                for cui in cui2tri_count:
                    if cui2tri_count[cui] > 0:
                        f.write(cui+'\n')

import numpy as np
import math
import json
import copy
from system.utils import softmax

rm3_paras = {'avg_doc_len':22.6, 'k1':1.2, 'k3':1.2, 'b':0.75, 'lambda':0.6, 'add_count':10}
w2idf = json.load(open('../release/mode/word2idf.local.json'))
q_json = json.load(open('../release/mode/qs2txt_idx.json'))
d_json = json.load(open('../release/mode/q2d2txt_idx.json'))

def BM25(q_dic, d_dic):
    global rm3_paras, w2idf
    doc_len = np.sum([item for item in d_dic.values()])
    bm25_score = 0
    for w in q_dic.keys():
        if w in d_dic.keys():
            sq = (rm3_paras['k3'] + 1) * q_dic[w] / (rm3_paras['k3'] + q_dic[w])
            K = rm3_paras['k1'] * (1 - rm3_paras['b'] + rm3_paras['b'] * doc_len / rm3_paras['avg_doc_len'])
            sd = (rm3_paras['k1'] + 1) * d_dic[w] / (K + d_dic[w])
            bm25_score += sd * sq * w2idf[w]
    return bm25_score

def LM(q_dic, d_dic):
    global rm3_paras, w2idf
    lm_score = 0
    for w in q_dic.keys():
        if w in d_dic.keys():
            lm_score += d_dic[w] * w2idf[w]
    return lm_score

def rm3_expansion(q, d_list, estimate_list):
    global d_json, q_json, rm3_paras, w2idf
    def add2dic(re_dic, w, v):
        if w in re_dic.keys():
            re_dic[w] += v
        else:
            re_dic[w] = v
    # get add words with current docs
    word2rel = {}
    estimate_list = softmax(estimate_list)
    for j in range(0, len(d_list)):
        pm = estimate_list[j]
        doc_len = np.sum([item for item in d_json[q][d_list[j]].values()])
        pq = 1
        for w in q_json[str(q)].keys():
            if w in d_json[str(q)][str(d_list[j])].keys():
                pqi = rm3_paras['lambda'] * d_json[q][d_list[j]][w] / doc_len + (1 - rm3_paras['lambda']) / math.exp(w2idf[w])
            else:
                pqi = (1 - rm3_paras['lambda']) / math.exp(w2idf[w])
            pq *= pqi
        for w in d_json[q][d_list[j]].keys():
            pwm = rm3_paras['lambda'] * d_json[q][d_list[j]][w] / doc_len + (1 - rm3_paras['lambda']) / math.exp(w2idf[w])            
            add2dic(word2rel, w, pq * pm * pwm)
    word2rel_sorted = sorted(word2rel.items(), key = lambda v: v[1], reverse = True)
    new_q_p = copy.deepcopy(q_json[q])
    current_count = 0
    for w_ in word2rel_sorted:
        w = w_[0]
        if current_count == rm3_paras['add_count']:
            break
        if w not in new_q_p:
            new_q_p[w] = 1
            current_count += 1
    return new_q_p
    
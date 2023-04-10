import json
import copy
import argparse
import numpy as np
from system.utils import add_result, combine_array, print_result2, bert_qm_all
import random
import math
import torch
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
random.seed(2021)
np.seterr(divide='raise',invalid='raise')

parser = argparse.ArgumentParser()
# bert-qm parameters
parser.add_argument('-m',type=int, help= 'the length of bert split trunks', required=False, default = 20)
parser.add_argument('-alpha',type=float, help= 'the rate of using expansion documents',required=False, default = 1.0)
parser.add_argument('-kc',type=int, help= 'the number of trunks', required=False, default = 10)
parser.add_argument('-kd',type=int, help= 'the number of bert split docs', required=False, default = 10)
parser.add_argument('-eeg_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 0.5)
parser.add_argument('-click_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 1.0)
parser.add_argument('-pse_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 1.0)
parser.add_argument('-path',type=str, required=False, default = 'rdr_baseline.json')
parser.add_argument('-gd_evaluate',type=str, required=False, default = 'True')
args = parser.parse_args()

args.gd_evaluate = True if args.gd_evaluate == 'True' else False

u2info = json.load(open('../release/u2info.json'))
user_list = [u for u in u2info.keys() if u.startswith('2_') == False]

q2d2score = json.load(open(f'../release/mode/q2d2score.json')) # m = 20
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

result_dic = {'ndcg@1':[],'ndcg@3':[],'ndcg@5':[],'ndcg@10':[],'map':[]}
result_list = {'bqe(bs)':copy.deepcopy(result_dic),'random':copy.deepcopy(result_dic),'online':copy.deepcopy(result_dic),'bert':copy.deepcopy(result_dic),'bqe(c)':copy.deepcopy(result_dic),'bqe(bs+c)':copy.deepcopy(result_dic),'bqe(un)':copy.deepcopy(result_dic),'bm25':copy.deepcopy(result_dic),'rm3(un)':copy.deepcopy(result_dic),'rm3(bs)':copy.deepcopy(result_dic),'rm3(c)':copy.deepcopy(result_dic),'rm3(bs+c)':copy.deepcopy(result_dic),'sogou':copy.deepcopy(result_dic),'bqe(bs+c-s)':copy.deepcopy(result_dic),'bqe(bs-s)':copy.deepcopy(result_dic), 'bqe(c-s)':copy.deepcopy(result_dic), 'bqe(un-s)': copy.deepcopy(result_dic), 'bqe(gd)': copy.deepcopy(result_dic)}
selected_runner = list(result_list.keys()) 
selected_runner = ['bert']
# selected_runner = ['bqe(gd)']


rm3_paras = {'avg_doc_len':22.6, 'k1':1.5, 'k3':1.5, 'b':0.75, 'lambda':0.6, 'add_count':10} # k1 k2 in [1.2,2.0]

u2result_list = {}
for u in user_list:
    u2result_list[u] = {}

u2idx2score = {}
for u in user_list:
    u2idx2score[u] = json.load(open(f'../release/idx2eeg_score/{u}.json'))

u2mean_score = {}
u2std_score = {}
for u in u2idx2score.keys():
    u2mean_score[u] = np.mean(list(u2idx2score[u].values()))
    u2std_score[u] = np.std(list(u2idx2score[u].values()))
general_mean = np.mean(list(u2mean_score.values()))
general_std = np.mean(list(u2std_score.values()))

anno = json.load(open('../release/mode/anno.json'))
w2idf = json.load(open('../release/mode/word2idf.json'))
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

def rm3_expansion(q, d_list, estimate_list):
    global d_json, q_json, rm3_paras, w2idf
    def add2dic(re_dic, w, v):
        if w in re_dic.keys():
            re_dic[w] += v
        else:
            re_dic[w] = v
    # get add words with current docs
    word2rel = {}
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

para = [args.alpha, args.click_gamma, args.eeg_gamma]

def detect_bad_click(d2score):
    num = 0
    # re = []
    # for d in d2score.keys():
    #     if d2score[d][1] == 1:
    #         re.append(d2score[d][3])
    # if len(re) > 0:
    #     mean_d2score = np.max(re)
    #     num = len([item for item in re if item < mean_d2score])
    # else:
    #     num = 0
    for d in d2score.keys():
        if d2score[d][1] == 1 and d2score[d][3] <= 1:
            num += 1
    return num

for u in user_list: 
    u2result_list[u]['0_0_0'] = copy.deepcopy(result_list)
    for raw_q in u2info[u]['raw_q2info'].keys():
        q = u2info[u]['raw_q2info'][raw_q]['q']
        # 使用刺激文件中的doc_list
        doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
        intent = u2info[u]['raw_q2info'][raw_q]['intent']
        raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
        for raw_d in [str(raw_d)]:
            now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
            now_d_score = []
            for d in now_d_list:           
                now_d_score.append(int(intent) in anno[q][d]['anno'])
            now_d_score2 = []
            # use user annotation as gd
            interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
            # if len(now_d_score) < 2 or np.std(now_d_score) == 0:
            #     continue
            d2score = {}
            now_d_score2_dic = {}
            for d in interactions.keys():
                goon = False
                info_list = [0, 0, 0, 0]
                for item in interactions[d]:
                    if item['motion'] == 'serp':
                        goon = True
                        if str(item['idx']) in u2idx2score[u].keys() and info_list[0] == 0:
                            info_list[0] = u2idx2score[u][str(item['idx'])]
                        if item['d'] not in now_d_score2_dic.keys():
                            now_d_score2_dic[item['d']] = item['score']
                        info_list[2] = item['score']
                    elif item['motion'] == 'land':
                        if str(item['idx']) in u2idx2score[u].keys():
                            info_list[0] = u2idx2score[u][str(item['idx'])]
                        now_d_score2_dic[item['d']] = item['score']
                        info_list[3] = item['score']
                        if goon:
                            info_list[1] = 1
                        break
                if goon == False:
                    info_list[0] = general_mean
                d2score[interactions[d][0]['d']] = info_list
            
            for d in now_d_list:
                try:
                    if now_d_score2_dic[d] == 1 and d2score[d][1] == 1:
                        now_d_score2.append(0)
                    else:
                        now_d_score2.append(now_d_score2_dic[d])
                except:
                    # jiayudebug snippet start----------
                    inputs = ''
                    while inputs != 'continue':
                        try:
                            print(eval(inputs))
                        except Exception as e:
                            print(e)
                        inputs = input()
                    # jiayudebug snippet end-------------
            if args.gd_evaluate:
                now_d_score = [item - 1 for item in now_d_score2]
            if len(now_d_score) < 2 or np.std(now_d_score) == 0 or np.max(now_d_score) == 0:
                continue
                
            un_d2score = {}
            for d in interactions.keys():
                info_list = [0, 0]
                info_list[0] = general_mean * q2d2score[q][d]['score']
                info_list[1] = 0.5 * q2d2score[q][d]['score']
                un_d2score[interactions[d][0]['d']] = info_list
            future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
            
            bad_click = detect_bad_click(d2score)

            if 'bert' in selected_runner:
                add_result(now_d_score, [q2d2score[q][d]['score'] for d in now_d_list], u2result_list[u]['0_0_0']['bert'], prev_len = len(now_d_list), information  = {'c':float(np.sum([v[1] for v in d2score.values()])), 'click': float(np.sum([v[1] for v in d2score.values()])), 'd_len':len(now_d_list)})

if args.path !='':
    json.dump(u2result_list, open(f'../results/part3_rdr/{args.path}', 'w'))

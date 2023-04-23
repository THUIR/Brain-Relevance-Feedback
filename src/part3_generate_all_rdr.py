import json
import copy
import argparse
import numpy as np
from system.utils import add_result, combine_array
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
parser.add_argument('-path',type=str, required=False, default = 'rdr_all.json')
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
selected_runner = ['bqe(bs+c-s)']
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

para = [args.alpha, args.click_gamma, args.eeg_gamma]

def detect_bad_click(d2score):
    num = 0
    for d in d2score.keys():
        if d2score[d][1] == 1 and d2score[d][3] <= 1:
            num += 1
    return num

for u in user_list: 
    for raw_q in u2info[u]['raw_q2info'].keys():
        q = u2info[u]['raw_q2info'][raw_q]['q']
        doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
        intent = u2info[u]['raw_q2info'][raw_q]['intent']
        raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
        for raw_d in [str(raw_d)]:
            now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
            now_d_score = []
            for d in now_d_list:           
                now_d_score.append(int(intent) in anno[q][d]['anno'])
            now_d_score2 = []
            # use user annotation as ground truth
            interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
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
                    # for debugging
                    inputs = ''
                    while inputs != 'continue':
                        try:
                            print(eval(inputs))
                        except Exception as e:
                            print(e)
                        inputs = input()
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
            if 'bqe(bs+c-s)' in selected_runner:
                search_list = [0,1,2,3,4,5]
                for pse_gamma in search_list:
                    for click_gamma in search_list:
                        for eeg_gamma in search_list:
                            if f'{pse_gamma}_{click_gamma}_{eeg_gamma}' not in u2result_list[u].keys():
                                u2result_list[u][f'{pse_gamma}_{click_gamma}_{eeg_gamma}'] = copy.deepcopy(result_list)
                            args.pse_gamma, args.click_gamma, args.eeg_gamma = pse_gamma, click_gamma, eeg_gamma
                            future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                            future_score2 = [pse_gamma * q2d2score[q][d]['score'] + args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for d in now_d_list]
                            add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u][f'{pse_gamma}_{click_gamma}_{eeg_gamma}']['bqe(bs+c-s)'], prev_len = len(now_d_list),information  = {'c':bad_click, 'click': float(np.sum([v[1] for v in d2score.values()])), 'd_len':len(now_d_list)})

if args.path !='':
    json.dump(u2result_list, open(f'../results/part3_rdr/{args.path}', 'w'))

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
parser.add_argument('-path',type=str, required=False, default = 'rdr_para_search.json')
parser.add_argument('-mode',type=str, required=False, default = '')
args = parser.parse_args()
if args.mode != '':
    args.path = args.path.split('.')
    args.path = args.path[0] + '.' + args.mode + '.' + args.path[1]

u2info = json.load(open('../release/u2info.json'))
user_list = [u for u in u2info.keys() if u.startswith('2_') == False]

q2d2score = json.load(open(f'../release/mode/q2d2score.json')) # m = 20
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

result_dic = {'ndcg@1':[],'ndcg@3':[],'ndcg@5':[], 'ndcg@10':[], 'map':[]}
result_list = {'bqe(bs)':copy.deepcopy(result_dic),'random':copy.deepcopy(result_dic),'online':copy.deepcopy(result_dic),'bert':copy.deepcopy(result_dic),'bqe(c)':copy.deepcopy(result_dic),'bqe(bs+c)':copy.deepcopy(result_dic),'bqe(un)':copy.deepcopy(result_dic),'bm25':copy.deepcopy(result_dic),'rm3(un)':copy.deepcopy(result_dic),'rm3(bs)':copy.deepcopy(result_dic),'rm3(c)':copy.deepcopy(result_dic),'rm3(bs+c)':copy.deepcopy(result_dic),'sogou':copy.deepcopy(result_dic),'bqe(bs+c-s)':copy.deepcopy(result_dic),'bqe(bs-s)':copy.deepcopy(result_dic), 'bqe(c-s)':copy.deepcopy(result_dic), 'bqe(un-s)': copy.deepcopy(result_dic)}
selected_runner = list(result_list.keys()) 
# selected_runner = ['bm25', 'rm3(un)', 'rm3(bs)', 'rm3(bs+c)', 'rm3(c)']
selected_runner = ['bqe(bs+c)',]

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
            
            # use user annotation as gd
            interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
            d2score = {}
            now_d_score2_dic = {}
            for d in interactions.keys():
                goon = False
                info_list = [0, 0]
                for item in interactions[d]:
                    if item['motion'] == 'serp':
                        goon = True
                        if str(item['idx']) in u2idx2score[u].keys() and info_list[0] == 0:
                            info_list[0] = u2idx2score[u][str(item['idx'])]
                        if item['d'] not in now_d_score2_dic.keys():
                            now_d_score2_dic[item['d']] = item['score']
                    elif item['motion'] == 'land':
                        if str(item['idx']) in u2idx2score[u].keys():
                            info_list[0] = u2idx2score[u][str(item['idx'])]
                        now_d_score2_dic[item['d']] = item['score']
                        if goon:
                            info_list[1] = 1
                        break
                if goon == False:
                    info_list[0] = general_mean
                d2score[interactions[d][0]['d']] = info_list
            for d in now_d_list:
                if now_d_score2_dic[d] == 1 and d2score[d][1] == 1:
                    now_d_score2.append(0)
                else:
                    now_d_score2.append(now_d_score2_dic[d])
            now_d_score = [item - 1 for item in now_d_score2]
            
            if len(now_d_score) < 2 or np.std(now_d_score) == 0 or np.max(now_d_score) == 0:
                continue
            
            future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
          
            for eeg_gamma in [0,1/5,1/3,1/4,1/2,1,2,3,4,5]:
                for click_gamma in [0,1/5,1/3,1/4,1/2,1,2,3,4,5]:
                    if click_gamma != 0 and eeg_gamma != 0:
                        if click_gamma / eeg_gamma > 5 or eeg_gamma / click_gamma > 5:
                            continue
                    args.click_gamma, args.eeg_gamma = click_gamma, eeg_gamma
                    if args.mode == '':
                        if 'bqe(bs+c)' in selected_runner:
                            future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                            future_score2 = [args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for d in now_d_list]
                            paras = '_'.join([str(item) for item in [args.alpha, args.click_gamma, args.eeg_gamma]])
                            if paras not in u2result_list[u].keys():
                                u2result_list[u][paras] = copy.deepcopy(result_list)
                            add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u][paras]['bqe(bs+c)'], prev_len = len(now_d_list), )
                    elif args.mode == 'bs':
                        future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                        future_score2 = [args.eeg_gamma * d2score[d][0] for d in now_d_list]
                        paras = '_'.join([str(item) for item in [args.alpha, args.click_gamma, args.eeg_gamma]])
                        if paras not in u2result_list[u].keys():
                            u2result_list[u][paras] = copy.deepcopy(result_list)
                        add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u][paras]['bqe(bs+c)'], prev_len = len(now_d_list), )
                    elif args.mode == 'c':
                        future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                        future_score2 = [args.click_gamma * d2score[d][1] for d in now_d_list]
                        paras = '_'.join([str(item) for item in [args.alpha, args.click_gamma, args.eeg_gamma]])
                        if paras not in u2result_list[u].keys():
                            u2result_list[u][paras] = copy.deepcopy(result_list)
                        add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u][paras]['bqe(bs+c)'], prev_len = len(now_d_list), )
                        

if args.path !='':
    json.dump(u2result_list, open(f'../results/part3_rdr/{args.path}', 'w'))

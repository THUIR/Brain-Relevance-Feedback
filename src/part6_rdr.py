import json
import copy
import argparse
import numpy as np
from system.utils import add_result, combine_array, print_result2
import random
from part3_bm25 import BM25, q_json, d_json
from scipy import stats
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
parser.add_argument('-eeg_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 5) # 0.5
parser.add_argument('-click_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 2.0) # 1
parser.add_argument('-path',type=str, required=False, default = 'part3_rdr.json')
parser.add_argument('-add_gamma',type=float, required=False, default = 0.5)
parser.add_argument('-gd_evaluate',type=str, required=False, default = 'True')
parser.add_argument('-para',type=str, help='the combination para', required=False, default = 'exp_all.gd')
parser.add_argument('-online',type=str, help='online testing experiments or offline testing experiments', required=False, default = 'False')
args = parser.parse_args()
args.online = True if args.online == 'True' else False
args.gd_evaluate = True if args.gd_evaluate == 'True' else False

u2info = json.load(open('../release/u2info.json'))
user_list = [u for u in u2info.keys() if u.startswith('2_') == args.online]

q2d2score = json.load(open(f'../release/mode/q2d2score.json')) # m = 20
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

result_dic = {'ndcg@1':[],'ndcg@3':[],'ndcg@5':[],'ndcg@10':[],'map':[]}
result_list = {'bqe(bs)':copy.deepcopy(result_dic),'random':copy.deepcopy(result_dic),'online':copy.deepcopy(result_dic),'bert':copy.deepcopy(result_dic),'bqe(c)':copy.deepcopy(result_dic),'bqe(bs+c)':copy.deepcopy(result_dic),'bqe(un)':copy.deepcopy(result_dic),'bm25':copy.deepcopy(result_dic),'rm3(un)':copy.deepcopy(result_dic),'rm3(bs)':copy.deepcopy(result_dic),'rm3(c)':copy.deepcopy(result_dic),'rm3(bs+c)':copy.deepcopy(result_dic),'sogou':copy.deepcopy(result_dic),'bqe(bs+c-s)':copy.deepcopy(result_dic),'bqe(bs-s)':copy.deepcopy(result_dic), 'bqe(c-s)':copy.deepcopy(result_dic), 'bqe(un-s)': copy.deepcopy(result_dic), 'bqe(gd)': copy.deepcopy(result_dic)}
selected_runner = list(result_list.keys()) 
selected_runner = ['bqe(bs+c)', 'bqe(bs+c-s)','bqe(c)','bqe(c-s)']

u2result_list = {}
for u in user_list:
    u2result_list[u] = copy.deepcopy(result_list)

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

# q2subset2para = json.load(open('../results/para/rdr.json'))
q2subset2para = json.load(open(f'../results/simulating/q2subset2click2para.rdr.{args.para}.json'))

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
                try:
                    if now_d_score2_dic[d] == 1 and d2score[d][1] == 1:
                        now_d_score2.append(0)
                    else:
                        now_d_score2.append(now_d_score2_dic[d])
                except:
                    # debuging
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
            num_click = np.sum([item[1] for item in d2score.values()])
            try:
                dic_para = q2subset2para[q]
                for ddd in sorted(now_d_list):
                    dic_para = dic_para[ddd]
                dic_para = dic_para['para']
                if type(dic_para) == dict:
                    if str(num_click) in dic_para.keys():
                        dic_para = dic_para[str(num_click)]
                    else:
                        dic_para = para
            except:
                dic_para = para
                
            un_d2score = {}
            for d in interactions.keys():
                info_list = [0, 0]
                info_list[0] = general_mean * q2d2score[q][d]['score']
                info_list[1] = 0.5 * q2d2score[q][d]['score']
                un_d2score[interactions[d][0]['d']] = info_list
            future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
            
            if 'bert' in selected_runner:
                add_result(now_d_score, [q2d2score[q][d]['score'] for d in now_d_list], u2result_list[u]['bert'], prev_len = len(now_d_list))
            if 'sogou' in selected_runner:
                add_result(now_d_score, [-int(d) for d in now_d_list], u2result_list[u]['sogou'], prev_len = len(now_d_list))

            args.alpha, args.click_gamma, args.eeg_gamma = dic_para
            args.eeg_gamma *= args.add_gamma
            if 'bqe(gd)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [(args.eeg_gamma+args.click_gamma) * (now_d_score2_dic[d] - 1)/3  for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(gd)'], prev_len = len(now_d_list))
            if 'bqe(bs)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [(args.eeg_gamma) * d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs)'], prev_len = len(now_d_list))
            if 'bqe(c)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [(args.click_gamma) * d2score[d][1] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(c)'], prev_len = len(now_d_list))
            if 'bqe(bs+c)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs+c)'], prev_len = len(now_d_list))
            if 'bqe(un)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.click_gamma * un_d2score[d][1] + args.eeg_gamma * un_d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(un)'], prev_len = len(now_d_list))

            args.alpha, args.click_gamma, args.eeg_gamma = para
            if 'bqe(bs-s)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.eeg_gamma * d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs-s)'], prev_len = len(now_d_list))
            if 'bqe(c-s)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.click_gamma * d2score[d][1] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(c-s)'], prev_len = len(now_d_list))
            if 'bqe(bs+c-s)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs+c-s)'], prev_len = len(now_d_list))
            if 'bqe(un-s)' in selected_runner:
                future_score1 = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score2 = [args.click_gamma * un_d2score[d][1] + args.eeg_gamma * un_d2score[d][0] for d in now_d_list]
                add_result(now_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(un-s)'], prev_len = len(now_d_list))
            
            bm25_score = [BM25(q_json[str(q)], d_json[str(q)][str(d)]) for d in now_d_list]
            if 'bm25' in selected_runner:
                add_result(now_d_score, bm25_score, u2result_list[u]['bm25'], prev_len = len(now_d_list))

for u in u2result_list.keys():
    for method in selected_runner:
        for key in result_list[method].keys():
            result_list[method][key] += u2result_list[u][method][key]

for method in selected_runner:
    print_result2(method, result_list[method])

method2metric2auc_list = {}
for u in list(u2result_list.keys()):
    for method in selected_runner:
        if method not in method2metric2auc_list.keys():
            method2metric2auc_list[method] = {}
        metric2auc_list = method2metric2auc_list[method]
        for key in result_list[method].keys():
            if key not in metric2auc_list.keys():
                metric2auc_list[key] = []
            metric2auc_list[key] += u2result_list[u][method][key]
if 'bqe(bs+c-s)' in selected_runner and 'bqe(c)' in selected_runner:
    for key in ['ndcg@1', 'ndcg@3', 'ndcg@5', 'map']:
        print(f'bqe(bs+c-s)>bqe(c-s), {key}: ', stats.ttest_rel(method2metric2auc_list['bqe(bs+c-s)'][key], method2metric2auc_list['bqe(c-s)'][key]))

if args.path !='':
    json.dump(u2result_list, open(f'../results/part3_rdr/{args.path}', 'w'))

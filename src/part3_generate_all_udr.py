import json
import copy
import argparse
import numpy as np
from system.utils import add_result, combine_array, bert_qm_all
import random
import copy
import tqdm
import torch
from transformers import AutoTokenizer,BertForSequenceClassification
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
random.seed(2021)
np.seterr(divide='raise',invalid='raise')


parser = argparse.ArgumentParser()
parser.add_argument('-alpha',type=float, help= 'the rate of using expansion documents',required=False, default = 1.0)
parser.add_argument('-kc',type=int, help= 'the number of trunks', required=False, default = 10)
parser.add_argument('-kd',type=int, help= 'the number of bert split docs', required=False, default = 10)
parser.add_argument('-eeg_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 3.0)
parser.add_argument('-click_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 4.5)
parser.add_argument('-path',type=str, help='the combination weight of supervised signal eeg', required=False, default = 'udr_all.json')
args = parser.parse_args()

para = [args.alpha, args.click_gamma, args.eeg_gamma]

u2info = json.load(open('../release/u2info.json'))
user_list = [u for u in u2info.keys() if u.startswith('2_') == False]

q2d2score = json.load(open(f'../release/mode/q2d2score.json')) # m = 20
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

result_dic = {'ndcg@1':[], 'ndcg@3':[], 'ndcg@5':[], 'ndcg@10':[], 'map':[]}
result_list = {'bqe(bs+c)':copy.deepcopy(result_dic),'random':copy.deepcopy(result_dic),'bqe(bs)':copy.deepcopy(result_dic),'online':copy.deepcopy(result_dic),'bert':copy.deepcopy(result_dic), 'bqe(c)':copy.deepcopy(result_dic), 'bqe(un)':copy.deepcopy(result_dic), 'bm25':copy.deepcopy(result_dic),'rm3(un)':copy.deepcopy(result_dic),'rm3(bs)':copy.deepcopy(result_dic),'rm3(c)':copy.deepcopy(result_dic),'rm3(bs+c)':copy.deepcopy(result_dic),'sogou':copy.deepcopy(result_dic),'lm':copy.deepcopy(result_dic),'brm3(un)':copy.deepcopy(result_dic), 'brm3(bs)':copy.deepcopy(result_dic), 'brm3(bs+c)':copy.deepcopy(result_dic), 'brm3(c)':copy.deepcopy(result_dic), 'bqe(bs+c-s)':copy.deepcopy(result_dic),'bqe(bs-s)':copy.deepcopy(result_dic), 'bqe(c-s)':copy.deepcopy(result_dic),'bqe(un-s)':copy.deepcopy(result_dic),'brm3(bs+c-s)':copy.deepcopy(result_dic),'bqe(gd-s)':copy.deepcopy(result_dic),'bqe(gd)':copy.deepcopy(result_dic)}
# please selected runners indicating what models do you want to run, where bqe(a+b) indicates query expansion using signals a, b, and pseudo-relevance signals, -s indicates using fixed parameter 
selected_runner = ['bqe(bs+c)', 'bert', 'bqe(c)', 'bqe(bs)', 'bqe(un)', 'sogou', 'bqe(bs-s)', 'bqe(c-s)', 'bqe(bs+c-s)'] 

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

idx2word = json.load(open('../release/mode/idx2word.json'))

def goon(u2info, u, u2result_list, device):
    if 'brm3(un)' in selected_runner or 'brm3(bs)' in selected_runner or 'brm3(bs+c)' in selected_runner:
        ranker = BertForSequenceClassification.from_pretrained('../resource/swh-checkpoint-1500')
        tokenizer = AutoTokenizer.from_pretrained('../resource/chinese_bert_wwm', use_fast=True)
        ranker.to(device)
        
    for raw_q in tqdm.tqdm(u2info[u]['raw_q2info'].keys()):
        q = u2info[u]['raw_q2info'][raw_q]['q']
        doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
        for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys():
            future_d_score = []
            future_d_list = []
            now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
            intent = u2info[u]['raw_q2info'][raw_q]['intent']
            for d in doc_list:
                if d not in now_d_list:
                    future_d_list.append(d)               
                    future_d_score.append(int(intent) in anno[q][d]['anno'])
            interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
            if len(future_d_score) < 2 or np.std(future_d_score) == 0:
                continue
            
            d2score = {}
            now_d_score2_dic = {}
            for d in interactions.keys():
                goon = False
                info_list = [0, 0]
                for item in interactions[d]:
                    if item['motion'] == 'serp':
                        goon = True
                        if str(item['idx']) in u2idx2score[u].keys():
                            info_list[0] = u2idx2score[u][str(item['idx'])]    
                        if item['d'] not in now_d_score2_dic.keys():
                            now_d_score2_dic[item['d']] = item['score']
                    elif item['motion'] == 'land':
                        now_d_score2_dic[item['d']] = item['score']
                        if goon:
                            info_list[1] = 1
                        break
                if goon == False:
                    info_list[0] = general_mean
                d2score[interactions[d][0]['d']] = info_list
            for d in now_d_score2_dic.keys():
                now_d_score2_dic[d] = [(now_d_score2_dic[d] - 1)/3, (now_d_score2_dic[d] - 1)/3]

            # use all parameters
            un_d2score = {}
            for d in interactions.keys():
                info_list = [0, 0]
                info_list[0] = general_mean * q2d2score[q][d]['score']
                info_list[1] = 0.5 * q2d2score[q][d]['score']
                un_d2score[interactions[d][0]['d']] = info_list
            
            if 'bqe(bs+c-s)' in selected_runner:
                search_list = [0,1,2,3,4,5]
                for pse_gamma in search_list:
                    for click_gamma in search_list:
                        for eeg_gamma in search_list:
                            if f'{pse_gamma}_{click_gamma}_{eeg_gamma}' not in u2result_list[u].keys(): 
                                u2result_list[u][f'{pse_gamma}_{click_gamma}_{eeg_gamma}'] = copy.deepcopy(result_list)
                            args.pse_gamma, args.click_gamma, args.eeg_gamma = pse_gamma, click_gamma, eeg_gamma
                            future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score)
                            add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u][f'{pse_gamma}_{click_gamma}_{eeg_gamma}']['bqe(bs+c-s)'], prev_len = len(now_d_list), information  = {'c':float(np.sum([v[1] for v in d2score.values()])), 'd_len':len(now_d_list)})
            
                
    return u2result_list[u]
def print_error(e):
    print('error', e)

multi_thread = True
if multi_thread:
    from multiprocessing.pool import Pool
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    pool = Pool(40)
    u2task = {}
    max_threads = 16
    tmp_threads = 0
    device_list = [1,2,6,8]
    for u in user_list:
        u2task[u] = (pool.apply_async(goon, args = (u2info, u, u2result_list, torch.device(f'cuda:{device_list[tmp_threads%len(device_list)]}')), error_callback = print_error))
        tmp_threads += 1
        if tmp_threads == max_threads:
            pool.close()
            pool.join()
            for u in u2task.keys():
                u2result_list[u] = u2task[u].get()
            pool = Pool(40)
            tmp_threads = 0
            u2task = {}
    pool.close()
    pool.join()
    for u in u2task.keys():
        u2result_list[u] = u2task[u].get()
else:
    for u in user_list:
        goon(u2info, u, u2result_list, torch.device(f'cuda:4'))

if args.path != '':
    json.dump(u2result_list, open(f'../results/part3_udr/{args.path}', 'w'))

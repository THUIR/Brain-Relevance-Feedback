import json
import copy
import argparse
import numpy as np
from system.utils import add_result, combine_array, print_result2, bert_qm_all
import random
import copy
import tqdm
import sklearn.preprocessing
import torch
from transformers import AutoTokenizer,BertForSequenceClassification
from part3_bm25 import BM25, rm3_expansion, q_json, d_json, LM
from part3_bert_fucntions import calc_score
np.random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
random.seed(2021)
np.seterr(divide='raise',invalid='raise') 

parser = argparse.ArgumentParser()
parser.add_argument('-alpha',type=float, help= 'the rate of using expansion documents',required=False, default = 1.0)
parser.add_argument('-kc',type=int, help= 'the number of trunks', required=False, default = 10)
parser.add_argument('-kd',type=int, help= 'the number of bert split docs', required=False, default = 10)
parser.add_argument('-eeg_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 3) # 3.0
parser.add_argument('-click_gamma',type=float, help='the combination weight of supervised signal eeg', required=False, default = 1) # 4.5
parser.add_argument('-path',type=str, help='the combination weight of supervised signal eeg', required=False, default = 'part3_udr.json')
parser.add_argument('-para',type=str, help='the combination para', required=False, default = 'exp_all.gd')
parser.add_argument('-online',type=str, help='online testing experiments or offline testing experiments', required=False, default = 'False')
args = parser.parse_args()
args.online = True if args.online == 'True' else False

para = [args.alpha, args.click_gamma, args.eeg_gamma]

u2info = json.load(open('../release/u2info.json'))
user_list = [u for u in u2info.keys() if u.startswith('2_') == args.online]

q2d2score = json.load(open(f'../release/mode/q2d2score.json')) # m = 20
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

result_dic = {'ndcg@1':[], 'ndcg@3':[], 'ndcg@5':[], 'ndcg@10':[], 'map':[]}
result_list = {'bqe(bs+c)':copy.deepcopy(result_dic),'random':copy.deepcopy(result_dic),'bqe(bs)':copy.deepcopy(result_dic),'online':copy.deepcopy(result_dic),'bert':copy.deepcopy(result_dic), 'bqe(c)':copy.deepcopy(result_dic), 'bqe(un)':copy.deepcopy(result_dic), 'bm25':copy.deepcopy(result_dic),'rm3(un)':copy.deepcopy(result_dic),'rm3(bs)':copy.deepcopy(result_dic),'rm3(c)':copy.deepcopy(result_dic),'rm3(bs+c)':copy.deepcopy(result_dic),'sogou':copy.deepcopy(result_dic),'lm':copy.deepcopy(result_dic),'brm3(un)':copy.deepcopy(result_dic), 'brm3(bs)':copy.deepcopy(result_dic), 'brm3(bs+c)':copy.deepcopy(result_dic), 'brm3(c)':copy.deepcopy(result_dic), 'bqe(bs+c-s)':copy.deepcopy(result_dic),'bqe(bs-s)':copy.deepcopy(result_dic), 'bqe(c-s)':copy.deepcopy(result_dic),'bqe(un-s)':copy.deepcopy(result_dic),'brm3(bs+c-s)':copy.deepcopy(result_dic),'bqe(gd-s)':copy.deepcopy(result_dic),'bqe(gd)':copy.deepcopy(result_dic)}
selected_runner = ['bqe(bs+c)', 'bert', 'bqe(c)', 'bqe(bs+c-s)'] 

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
# anno = json.load(open('tmp_anno.json'))

# q2subset2para = json.load(open('../results/para/udr.json'))
# q2subset2para = json.load(open('../results/simulating/q2subset2para.exp_all.gd.json'))
q2subset2para = json.load(open(f'../results/simulating/q2subset2click2para.{args.para}.json'))

idx2word = json.load(open('/home/yzy/resource/idx2word.json'))

def goon(u2info, u, u2result_list, device):
    if 'brm3(un)' in selected_runner or 'brm3(bs)' in selected_runner or 'brm3(bs+c-s)' in selected_runner:
        ranker = BertForSequenceClassification.from_pretrained('/home/yzy/resource/swh-checkpoint-1500')
        tokenizer = AutoTokenizer.from_pretrained('/home/yzy/resource/chinese_bert_wwm', use_fast=True)
        ranker.to(device)
    for raw_q in tqdm.tqdm(u2info[u]['raw_q2info'].keys()):
        q = u2info[u]['raw_q2info'][raw_q]['q']
        doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
        # doc_list = q2doc_list[q]
        for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys():
            future_d_score = []
            future_d_list = []
            now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
            intent = u2info[u]['raw_q2info'][raw_q]['intent']
            for d in doc_list:
                if d not in now_d_list:
                    future_d_list.append(d)
                    try:               
                        future_d_score.append(int(intent) in anno[q][d]['anno'])
                    except:
                        future_d_score.append(0)
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
            
            num_click = np.sum([item[1] for item in d2score.values()])

            ddd = '_'.join(sorted(now_d_list))
            if len(now_d_list) <= 3:
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
                    print('para not found')
                    dic_para =  para
            else:
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
                    # print('error')
                    # # jiayudebug snippet start----------
                    # inputs = ''
                    # while inputs != 'continue':
                    #     try:
                    #         print(eval(inputs))
                    #     except Exception as e:
                    #         print(e)
                    #     inputs = input()
                    # # jiayudebug snippet end-------------
                    dic_para =  para

            # use all parameters?
            un_d2score = {}
            for d in interactions.keys():
                info_list = [0, 0]
                info_list[0] = general_mean * q2d2score[q][d]['score']
                info_list[1] = 0.5 * q2d2score[q][d]['score']
                un_d2score[interactions[d][0]['d']] = info_list
            

            if 'bqe(bs+c)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs+c)'], prev_len = len(now_d_list))
            if 'bqe(bs+c-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs+c-s)'], prev_len = len(now_d_list))
            if 'bqe(gd-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, now_d_score2_dic)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(gd-s)'], prev_len = len(now_d_list))
            if 'bqe(gd)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, now_d_score2_dic)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(gd)'], prev_len = len(now_d_list))
            if 'bqe(bs)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                args_copy = copy.deepcopy(args)
                args_copy.click_gamma = 0
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args_copy, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs)'], prev_len = len(now_d_list))
            if 'bqe(bs-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                args_copy = copy.deepcopy(args)
                args_copy.click_gamma = 0
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args_copy, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(bs-s)'], prev_len = len(now_d_list))
            if 'bqe(c)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                args_copy = copy.deepcopy(args)
                args_copy.eeg_gamma = 0
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args_copy, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(c)'], prev_len = len(now_d_list))
            if 'bqe(c-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                args_copy = copy.deepcopy(args)
                args_copy.eeg_gamma = 0
                # args_copy.click_gamma += args.eeg_gamma
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args_copy, q2d2d2score, d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(c-s)'], prev_len = len(now_d_list))
            if 'bqe(un)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, un_d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(un)'], prev_len = len(now_d_list))
            if 'bqe(un-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, un_d2score)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['bqe(un-s)'], prev_len = len(now_d_list))
            if 'online' in selected_runner:
                add_result(future_d_score, [-i for i in range(len(future_d_list))], u2result_list[u]['online'], prev_len = len(now_d_list))
            if 'random' in selected_runner:
                random_future_d_list = []
                random_future_d_score = []
                for d in q2d2score[q].keys():
                    if d not in now_d_list:
                        random_future_d_list.append(d)               
                        random_future_d_score.append(int(intent) in anno[q][d]['anno'])
                add_result(random_future_d_score, [random.random() for i in range(len(random_future_d_list))], u2result_list[u]['random'], prev_len = len(now_d_list))
            if 'bert' in selected_runner:
                add_result(future_d_score, [q2d2score[q][d]['score'] for d in future_d_list], u2result_list[u]['bert'], prev_len = len(now_d_list))
            if 'sogou' in selected_runner:
                add_result(future_d_score, [-int(d) for d in future_d_list], u2result_list[u]['sogou'], prev_len = len(now_d_list))
            if 'bm25' in selected_runner:
                bm25_now_score = sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([BM25(q_json[str(q)], d_json[str(q)][str(d)]) for d in now_d_list],1)).squeeze(1)
                future_score1 = sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([BM25(q_json[str(q)], d_json[str(q)][str(d)]) for d in future_d_list],1)).squeeze(1)
                add_result(future_d_score, future_score1, u2result_list[u]['bm25'], prev_len = len(now_d_list))
            if 'lm' in selected_runner:
                lm_future_score1 = sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([LM(q_json[str(q)], d_json[str(q)][str(d)]) for d in future_d_list],1)).squeeze(1)
                add_result(future_d_score, lm_future_score1, u2result_list[u]['lm'], prev_len = len(now_d_list))
            
            args.alpha, args.click_gamma, args.eeg_gamma = para
            if 'rm3(un)' in selected_runner:
                future_score2 = [bm25_now_score[i] + args.click_gamma * un_d2score[d][1] + args.eeg_gamma * un_d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = [BM25(q_exp, d_json[str(q)][str(d)]) for d in future_d_list]
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['rm3(un)'], prev_len = len(now_d_list))
            if 'rm3(c)' in selected_runner:
                future_score2 = [bm25_now_score[i] + args.click_gamma * d2score[d][1] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = [BM25(q_exp, d_json[str(q)][str(d)]) for d in future_d_list]
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['rm3(c)'], prev_len = len(now_d_list))
            if 'rm3(bs)' in selected_runner:
                future_score2 = [bm25_now_score[i] + args.eeg_gamma * d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = [BM25(q_exp, d_json[str(q)][str(d)]) for d in future_d_list]
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['rm3(bs)'], prev_len = len(now_d_list))
            if 'rm3(bs+c)' in selected_runner:
                future_score2 = [bm25_now_score[i] + args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = [BM25(q_exp, d_json[str(q)][str(d)]) for d in future_d_list]
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['rm3(bs+c)'], prev_len = len(now_d_list))
            
            if 'brm3(bs+c)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = dic_para
                bert_score = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
                future_score2 = [bert_score[i] + args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = np.squeeze(sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([calc_score(tokenizer, ranker, q_exp, d_json[str(q)][str(d)], idx2word, device)[0].detach().cpu().numpy().tolist() for d in future_d_list], 1),), 1)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['brm3(bs+c)'], prev_len = len(now_d_list))
            if 'brm3(bs+c-s)' in selected_runner:
                args.alpha, args.click_gamma, args.eeg_gamma = para
                bert_score = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
                future_score2 = [bert_score[i] + args.click_gamma * d2score[d][1] + args.eeg_gamma * d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = np.squeeze(sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([calc_score(tokenizer, ranker, q_exp, d_json[str(q)][str(d)], idx2word, device)[0].detach().cpu().numpy().tolist() for d in future_d_list], 1),), 1)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['brm3(bs+c-s)'], prev_len = len(now_d_list))
            if 'brm3(bs)' in selected_runner:
                bert_score = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
                future_score2 = [bert_score[i] + args.eeg_gamma * d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = np.squeeze(sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([calc_score(tokenizer, ranker, q_exp, d_json[str(q)][str(d)], idx2word, device)[0].detach().cpu().numpy().tolist() for d in future_d_list], 1),), 1)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['brm3(bs)'], prev_len = len(now_d_list))
            if 'brm3(c)' in selected_runner:
                bert_score = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
                future_score2 = [bert_score[i] + args.click_gamma * d2score[d][1] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = np.squeeze(sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([calc_score(tokenizer, ranker, q_exp, d_json[str(q)][str(d)], idx2word, device)[0].detach().cpu().numpy().tolist() for d in future_d_list], 1),), 1)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['brm3(c)'], prev_len = len(now_d_list))
            if 'brm3(un)' in selected_runner:
                bert_score = [q2d2score[q][d]['score'] for d in now_d_list]
                future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
                future_score2 = [bert_score[i] + args.click_gamma * un_d2score[d][1] + args.eeg_gamma * un_d2score[d][0] for i,d in enumerate(now_d_list)]
                q_exp = rm3_expansion(q, now_d_list, future_score2)
                future_score2 = np.squeeze(sklearn.preprocessing.MinMaxScaler().fit_transform(np.expand_dims([calc_score(tokenizer, ranker, q_exp, d_json[str(q)][str(d)], idx2word, device)[0].detach().cpu().numpy().tolist() for d in future_d_list], 1),), 1)
                add_result(future_d_score, combine_array(future_score2, future_score1, args.alpha), u2result_list[u]['brm3(un)'], prev_len = len(now_d_list))
    return u2result_list[u]
def print_error(e):
    print('error', e)

multi_thread = False
if multi_thread:
    from multiprocessing.pool import Pool
    # import torch
    # torch.multiprocessing.set_start_method('spawn')
    pool = Pool(40)
    u2task = {}
    max_threads = 16
    tmp_threads = 0
    device_list = [6,7,8,9]
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

for u in list(u2result_list.keys()):
    for method in selected_runner:
        for key in result_list[method].keys():
            try:
                result_list[method][key] += (u2result_list[u][method][key])
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

for method in selected_runner:
    print_result2(method, result_list[method])

from scipy import stats
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

if 'bqe(bs+c)' in selected_runner and 'bqe(c)' in selected_runner:
    for key in ['ndcg@1', 'ndcg@3', 'ndcg@5', 'map']:
        print(f'bqe(bs+c)>bqe(c), {key}: ', stats.ttest_rel(method2metric2auc_list['bqe(bs+c)'][key], method2metric2auc_list['bqe(c)'][key]))


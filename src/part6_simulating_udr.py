import json
import argparse
import numpy as np
import copy 
from system.utils import bert_qm_all, add_result, combine_array, get_metric_sum
import tqdm
from sklearn.metrics import ndcg_score
import random
random.seed(2022)
np.random.seed(2022)
from part6_simulating_utils import Simulator, anno, q2d2score, true_data, q2doc_list, q2d2d2score, grouping, lm_grouping, roberta_grouping, get_true_data, roberta_grouping_soft
import copy
from multiprocessing.pool import Pool
import os
from sklearn.preprocessing import MinMaxScaler
cpu_num = 32 # 这里设置成你想运行的CPU个数
os.environ["OMP_NUM_THREADS"] = str(cpu_num)  # noqa
os.environ["MKL_NUM_THREADS"] = str(cpu_num) # noqa 

def run_query(query_info, idx, simulator, result_dic, q2d2score, q2d2d2score, anno, args,):
    intent2subset2para2performance = {}
    if type(query_info['intent']) == dict:
        total_intent = [int(intent) for intent in query_info['intent'].keys()]
    else:
        total_intent = [int(intent) for intent in query_info['intent']]
    
    q = str(query_info['q'])

    for intent in tqdm.tqdm(total_intent):
        if int(intent) == -1:
            continue
        intent2subset2para2performance[intent] = {} 
        for subset_idx, subset in enumerate(query_info['subset']):
            intent2subset2para2performance[intent][subset_idx] = {}
            future_d_score = []
            future_d_list = query_info['restset'][subset_idx]
            now_d_list = subset
            now_d_score = []

            if args.grouping_method == 'roberta_soft':
                for d in future_d_list: 
                    if anno[q][str(d)]['anno'] == -1:
                        future_d_score.append(-1e-6)
                    else:
                        future_d_score.append(anno[q][str(d)]['anno'][int(intent)])
                for d in now_d_list: 
                    if anno[q][str(d)]['anno'] == -1:
                        now_d_score.append(-1e-6)
                    else:
                        try:
                            now_d_score.append(anno[q][str(d)]['anno'][int(intent)])
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
                all_d_score = now_d_score + future_d_score
                all_d_score = [item for item in all_d_score if item != -1e-6]
                scaler = MinMaxScaler()
                scaler.fit(np.expand_dims(all_d_score, 1))
            else:
                for d in future_d_list:               
                    future_d_score.append(int(intent) in anno[q][str(d)]['anno'])

            if len(future_d_score) <= 1 or np.std(future_d_score) == 0:
                continue
            if args.grouping_method == 'roberta_soft':
                jie = np.squeeze(scaler.transform(np.expand_dims(all_d_score, 1)), 1)
                jie = sorted(jie)[int(len(jie)*0.75)]
            else:
                jie = 0

            for rand in range(20): # 100
                d2score = {}
                for d in now_d_list:
                    info_list = [0, 0]
                    if args.grouping_method == 'roberta_soft':
                        if anno[q][str(d)]['anno'] == -1:
                            truth = 0
                        else:
                            truth = scaler.transform(np.expand_dims([anno[q][str(d)]['anno'][int(intent)]], 1))[0][0]
                        info_list[0] = simulator.calc_score_anno2(300, truth)
                    else:
                        truth = int(intent) in anno[q][str(d)]['anno']
                        info_list[0] = simulator.calc_score_anno(300, truth)
                    info_list[1] = 0 if random.random() > args.click_rate else truth > jie
                    # 1028
                    info_list[1] = 1 if random.random() > args.click_rate else info_list[1]
                    # 如果是前三个文档，那没法点击
                    if len(now_d_list) <= 2:
                        info_list[1] = 0
                    d2score[d] = info_list
                num_click = np.sum([item[1] for item in d2score.values()])

                for eeg_gamma in [0,1/5,1/3,1/4,1/2,1,2,3,4,5]:
                    for click_gamma in [0,1/5,1/3,1/4,1/2,1,2,3,4,5]:
                        if click_gamma != 0 and eeg_gamma != 0:
                            if click_gamma / eeg_gamma > 5 or eeg_gamma / click_gamma > 5:
                                continue
                        args.click_gamma, args.eeg_gamma = click_gamma, eeg_gamma
                        future_score1, future_score2 = bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score)
                        for alpha in [-3,-2,-1,0,1.0,2.0,3.0]:
                            args.alpha = alpha
                            # based on ndcg@10 + ndcg@3
                            re = ndcg_score([future_d_score], [combine_array(future_score2, future_score1, args.alpha)], k=10)
                            if args.click == 'add':
                                para = f'{alpha}_{click_gamma}_{eeg_gamma}_{num_click}'
                            else:
                                para = f'{alpha}_{click_gamma}_{eeg_gamma}'
                            if para not in intent2subset2para2performance[intent][subset_idx].keys():
                                intent2subset2para2performance[intent][subset_idx][para] = []
                            intent2subset2para2performance[intent][subset_idx][para].append(float(re))
    return intent2subset2para2performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-grouping_method',type=str, help= 'the method that group documents w.r.t. their sub-topic', required=False, default = 'gd')
    parser.add_argument('-m',type=int, help= 'the length of bert split trunks', required=False, default = 20)
    parser.add_argument('-kc',type=int, help= 'the number of trunks', required=False, default = 10)
    parser.add_argument('-kd',type=int, help= 'the number of bert split docs', required=False, default = 10)
    parser.add_argument('-multi_thread',type=str, required=False, default = 'True', choices=['True', 'False'])
    parser.add_argument('-click_rate',type=float, required=False, default = 0.9)
    parser.add_argument('-mode',type=str, required=False, default = 'sample', choices=['exp', 'iter', 'sample', 'all', 'exp_all', 'exp_pre'])
    parser.add_argument('-click',type=str, required=False, default = 'no', choices=['no','add'])
    parser.add_argument('-task',type=str, required=False, default = 'udr')

    args = parser.parse_args()
    args.multi_thread = True if args.multi_thread == 'True' else False

    true_data = get_true_data(true_data, args, )
    simulator = Simulator()
    result_dic = {'ndcg@1':[],'ndcg@3':[],'ndcg@5':[],'ndcg@10':[],'map':[]}

    # clustering based on bert scores
    if args.grouping_method == 'bert':
        for i in range(len(true_data)):
            q = true_data[i]['q']
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            intent_list = true_data[i]['intent']
            clustering_results = grouping(intent_list, anno, q, q2d2d2score, d_list=d_list)
            q = true_data[i]['q']
            true_data[i]['intent'] = [int(item) for item in list(set(clustering_results))]
            for idx, d in enumerate(d_list):
                anno[q][str(d)]['anno'] = [clustering_results[idx]]
    if args.grouping_method == 'roberta':
        for i in range(len(true_data)):
            query_info = true_data[i]
            if type(query_info['intent']) == dict:
                total_intent = [int(intent) for intent in query_info['intent'].keys()]
            else:
                total_intent = [int(intent) for intent in query_info['intent']]
            q = true_data[i]['q']
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            intent_list = true_data[i]['intent']
            clustering_results = roberta_grouping(intent_list, anno, q, q2d2d2score, d_list=d_list, cluster_num = len(total_intent))
            q = true_data[i]['q']
            true_data[i]['intent'] = [int(item) for item in list(set(clustering_results))]
            for idx, d in enumerate(d_list):
                anno[q][str(d)]['anno'] = [clustering_results[idx]]
    elif args.grouping_method == 'roberta_soft':
        for i in range(len(true_data)):
            query_info = true_data[i]
            if type(query_info['intent']) == dict:
                total_intent = [int(intent) for intent in query_info['intent'].keys()]
            else:
                total_intent = [int(intent) for intent in query_info['intent']]
            q = true_data[i]['q']
            d_list = list(anno[q].keys())
            intent_list = true_data[i]['intent']
            clustering_results = roberta_grouping_soft(intent_list, anno, q, q2d2d2score, d_list=d_list, cluster_num = len(total_intent))
            q = true_data[i]['q']
            for item in list(clustering_results):
                if type(item)==dict:
                    true_data[i]['intent'] = list(item.keys())
                    break
            for idx, d in enumerate(d_list):
                anno[q][str(d)]['anno'] = clustering_results[idx]

    elif 'bert' in args.grouping_method:
        cluster_num = int(args.grouping_method[4:])
        for i in range(len(true_data)):
            q = true_data[i]['q']
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            intent_list = true_data[i]['intent']
            clustering_results = grouping(intent_list, anno, q, q2d2d2score, d_list=d_list, cluster_num = cluster_num)
            q = true_data[i]['q']
            true_data[i]['intent'] = [int(item) for item in list(set(clustering_results))]
            for idx, d in enumerate(d_list):
                anno[q][str(d)]['anno'] = [clustering_results[idx]]
    elif 'lm' == args.grouping_method:
        d_json = json.load(open('../release/mode/q2d2txt_idx.json'))
        for i in range(len(true_data)):
            q = true_data[i]['q']
            d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
            intent_list = true_data[i]['intent']
            clustering_results = lm_grouping(intent_list, anno, q, q2d2d2score, d_list=d_list, d_json=d_json, cluster_num = 5)
            q = true_data[i]['q']
            true_data[i]['intent'] = [int(item) for item in list(set(clustering_results))]
            for idx, d in enumerate(d_list):
                anno[q][str(d)]['anno'] = [clustering_results[idx]]

    if args.multi_thread:
        pool_num = 50
        pool_idx = 0
        pool = Pool(pool_num + 5)
        result_list = []
        task_list = []
        for i in range(len(true_data)):
            task_list.append(pool.apply_async(run_query, args = (true_data[i], i, simulator, result_dic, q2d2score, q2d2d2score, anno, args,)))
            pool_idx += 1
            if pool_idx == pool_num:
                pool.close()
                pool.join()
                for task in task_list:
                    result_list.append(task.get())
                pool_idx = 0
                pool = Pool(pool_num + 5)
                task_list = []
        pool.close()
        pool.join()
        for task in task_list:
            result_list.append(task.get())
        pool_idx = 0
        pool = Pool(pool_num + 5)
        task_list = []
    else:
        result_list = []
        for i in [1]: # tqdm.tqdm(range(len(true_data)))
            result_list.append(run_query(true_data[i], i, simulator, result_dic, q2d2score, q2d2d2score, anno, args,))

    json.dump(result_list, open(f'../results/simulating/true_data_result_list.{args.mode}.{args.grouping_method}.json','w'))
    json.dump(true_data, open(f'../results/simulating/true_data.{args.mode}.{args.grouping_method}.json','w'))


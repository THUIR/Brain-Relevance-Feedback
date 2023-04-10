import json
import numpy as np
import random
import sklearn.cluster
import torch
import argparse
random.seed(2022)
np.random.seed(2022)

def subsets(nums):
    result = [[]]
    for num in nums:
        for element in result[:]:
            x=element[:]
            x.append(num)
            result.append(x)        
    return result[1:-1]

import itertools
def findsubsets(s, n):
    return list(itertools.combinations(s, n))
    
def find_subset_dids(subset_ids, doc_list):
    re = []
    rest = []
    for i in range(len(doc_list)):
        if i in subset_ids:
            re.append(doc_list[i]['did'])
        else:
            rest.append(doc_list[i]['did'])
    return re, rest

def get_true_data(true_data, args):
    if args.task == 'rdr':
        u2info = json.load(open('../release/u2info.json'))
        q2sets = {}
        for u in u2info.keys():
            if u.startswith('2_'):
               continue 
            for raw_q in u2info[u]['raw_q2info'].keys():
                q = u2info[u]['raw_q2info'][raw_q]['q']
                if q not in q2sets.keys():
                    q2sets[q] = {'re':[], 'rest':[]}
                raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
                now_d_list = u2info[u]['raw_q2task2info'][raw_q][str(raw_d)]['now_d_list']
                q2sets[q]['re'] += [now_d_list]
                q2sets[q]['rest'] += [[]]
        true_data_tmp = []          
        q2true_data_i = {}  
        for i in range(len(true_data)):
            q = true_data[i]['q']
            q2true_data_i[q] = i
            if q in q2sets.keys():
                true_data[i]['subset'] = q2sets[q]['re']
                true_data[i]['restset'] = q2sets[q]['rest']
                true_data[i]['intent'] = []   
                true_data_tmp.append(true_data[i]) 
        true_data = true_data_tmp
        for u in u2info.keys():
            for raw_q in u2info[u]['raw_q2info'].keys():
                q=u2info[u]['raw_q2info'][raw_q]['q']
                intent = int(u2info[u]['raw_q2info'][raw_q]['intent'])
                if intent not in true_data[q2true_data_i[q]]['intent']:
                    true_data[q2true_data_i[q]]['intent'].append(intent)  
    elif 'exp' in args.mode:
        u2info = json.load(open('../release/u2info.json'))
        q2doc_list = {}
        for item in true_data:
            q2doc_list[item['q']] = [a['did'] for a in item['doc_list']]
        q2sets = {}
        for u in u2info.keys():
            if 'pre' in args.mode and u.startswith('2_'):
                continue
            
            if 'all' in args.mode or 'pre' in args.mode or u.startswith('2_'):
                for raw_q in u2info[u]['raw_q2info'].keys():
                    q = u2info[u]['raw_q2info'][raw_q]['q']
                    if q not in q2sets.keys():
                        q2sets[q] = {'re':[], 'rest':[]}
                    raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
                    now_d_list = u2info[u]['raw_q2task2info'][raw_q][str(raw_d)]['now_d_list']
                    re = []
                    rest = []
                    other_list = q2doc_list[q]
                    for iter_i in range(1, len(now_d_list) + 1): # len(now_d_list))
                        col = now_d_list[:iter_i]
                        re.append(col)
                        rest.append([])
                        for d in other_list:
                            if d not in re[-1]:
                                rest[-1].append(d)
                    q2sets[q]['re'] += re
                    q2sets[q]['rest'] += rest 
            
        true_data_tmp = []          
        q2true_data_i = {}  
        for i in range(len(true_data)):
            q = true_data[i]['q']
            q2true_data_i[q] = i
            if q in q2sets.keys():
                true_data[i]['subset'] = q2sets[q]['re']
                true_data[i]['restset'] = q2sets[q]['rest']
                true_data[i]['intent'] = []   
                true_data_tmp.append(true_data[i]) 
        true_data = true_data_tmp
        for u in u2info.keys():
            for raw_q in u2info[u]['raw_q2info'].keys():
                q=u2info[u]['raw_q2info'][raw_q]['q']
                intent = int(u2info[u]['raw_q2info'][raw_q]['intent'])
                if intent not in true_data[q2true_data_i[q]]['intent']:
                    true_data[q2true_data_i[q]]['intent'].append(intent)  
    else:
        for i in range(len(true_data)):
            if args.mode == 'iter':
                d2score = [[d, q2d2score[true_data[i]['q']][true_data[i]['doc_list'][d]['did']]] for d in range(len(true_data[i]['doc_list']))]
                best_d_i = max(d2score, key = lambda v:v[1]['score'],)[0]
                other_list = []
                for d_i in range(len(range(len(true_data[i]['doc_list'])))):
                    if d_i != best_d_i:
                        other_list.append(d_i)
                if best_d_i == -1:
                    print('error')
                    exit()
                
                best_d_i = true_data[i]['doc_list'][best_d_i]['did']
                other_list = [true_data[i]['doc_list'][item]['did'] for item in other_list]

                re = []
                rest = []

                for iter_i in range(3):
                    list_of_col = findsubsets(other_list, iter_i)
                    for col_ in list_of_col:
                        col = list(col_)
                        col.append(best_d_i)
                        re.append(col)
                        rest.append([])
                        for d in other_list:
                            if d not in re[-1]:
                                rest[-1].append(d)
                true_data[i]['subset'] = re
                true_data[i]['restset'] = rest            
            else:
                enumerate_list = list(range(len(true_data[i]['doc_list'])))
                subset_ids = subsets(enumerate_list)
                true_data[i]['subset_ids'] = subset_ids
                true_data[i]['subset'] = []
                true_data[i]['restset'] = []
            
                for subset_id in subset_ids:
                    re, rest = find_subset_dids(subset_id, true_data[i]['doc_list'])
                    true_data[i]['subset'].append(re)
                    true_data[i]['restset'].append(rest)
                if args.mode == 'sample':
                    re = []
                    rest = []
                    perm = np.random.permutation(len(true_data[i]['subset']))[:1000]
                    for perm_idx in perm:
                        re.append(true_data[i]['subset'][perm_idx])
                        rest.append(true_data[i]['restset'][perm_idx])
                    true_data[i]['subset'] = re
                    true_data[i]['restset'] = rest
    return true_data

class Simulator:
    def __init__(self, general_std = 1.3, decrease_rate=0.92, mod=100,):
        self.general_std = general_std
        self.decrease_rate = decrease_rate
        self.mod = mod
        self.para = {'anno':{}, 'gd':{}}
        self.para['anno'][0] = {'loc': 0.4294, 'scale': 0.1263}
        self.para['anno'][1] = {'loc': 0.47, 'scale': 0.1259}
        self.para['gd'][1] = {'loc': 0.4784, 'scale': 0.1317}
        self.para['gd'][0] = {'loc': 0.4210, 'scale': 0.1208}

    def calc_score(self, shot, y_true):
        y_true = 1 if y_true > 1 else 0
        scale = self.para['gd'][y_true]['scale'] * self.general_std * self.decrease_rate ** (shot / self.mod)
        return np.random.normal(self.para['gd'][y_true]['loc'], scale)

    def calc_score_anno(self, shot, y_true):
        scale = self.para['anno'][y_true]['scale'] * self.general_std * self.decrease_rate ** (shot / self.mod)
        return np.random.normal(self.para['anno'][y_true]['loc'], scale)

    def calc_score_anno2(self, shot, y_score):
        scale = 0.12 * self.general_std * self.decrease_rate ** (shot / self.mod)
        return np.random.normal(y_score * 0.9, scale)

anno = json.load(open('../release/mode/anno.json'))
q2d2score = json.load(open(f'../release/mode/q2d2score.json'))
q2d2d2score = json.load(open(f'../release/mode/q2d2d2score.json'))

true_data = json.load(open('../release/user_study_simulation_example.json'))
q2doc_list = {}
for item in true_data:
    q2doc_list[item['q']] = [a['did'] for a in item['doc_list']]

def grouping(intent_list, anno, q, q2d2d2score, d_list, cluster_num = 5):
    # 只去group那些有intent的
    d2vec = {}
    for d in d_list:
        d2vec[d] = np.zeros(len(d_list))
        for idx, d2 in enumerate(d_list):
            d2vec[d][idx] = q2d2d2score[q][d][d2]['score']
    vec_matrix = list(d2vec.values())
    kmeans = sklearn.cluster.KMeans(n_clusters = cluster_num, )
    y_kmeans = kmeans.fit_predict(vec_matrix)
    for i in range(len(d_list)):
        if len(set(anno[q][str(d)]['anno']) & set(intent_list)) == 0:
            y_kmeans[i] = -1
    return y_kmeans

def swh_format2dict(roberta):
    re = {}
    select_name = 'crop_text'
    vectors = roberta[0]
    ids = roberta[1]
    for i, file_name in enumerate(ids):
        dir, q, d = file_name.split('.')[0].split('-')
        if dir != select_name:
            continue
        if q not in re.keys():
            re[q] = {}
        re[q][d] = vectors[i].numpy().tolist()
    return re   

roberta = torch.load('/home/swh/BMI/projects/data/encoded_corpus/roberta.pt')
roberta = swh_format2dict(roberta)

def roberta_grouping(intent_list, anno, q, q2d2d2score, d_list, cluster_num = 5):
    global roberta
    # 只去group那些有intent的
    y_kmeans = np.zeros(len(d_list))
    d2vec = {}
    for i, d in enumerate(d_list):
        if len(set(anno[q][str(d)]['anno']) & set(intent_list)) == 0:
            y_kmeans[i] = -1
        else:
            d2vec[d] = roberta[q][d]
    vec_matrix = list(d2vec.values())
    try:
        kmeans = sklearn.cluster.KMeans(n_clusters = cluster_num, )
        y_kmeans2 = kmeans.fit_predict(vec_matrix)
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
    j = 0
    for i in range(len(d_list)):
        if y_kmeans[i] != -1:
            y_kmeans[i] = y_kmeans2[j]
            j += 1
    return y_kmeans

def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def roberta_grouping_soft(intent_list, anno, q, q2d2d2score, d_list, cluster_num = 5):
    global roberta
    # 只去group那些有intent的
    y_kmeans = [0 for i in range((len(d_list)))]
    d2vec = {}
    for i, d in enumerate(d_list):
        if len(set(anno[q][str(d)]['anno']) & set(intent_list)) == 0:
            y_kmeans[i] = -1
        else:
            d2vec[d] = roberta[q][d]
    vec_matrix = list(d2vec.values())
    kmeans = sklearn.cluster.KMeans(n_clusters = cluster_num, )
    
    y_kmeans2 = [dict() for i in range(len(vec_matrix))]
    kmeans.fit(vec_matrix)
    for i,center in enumerate(kmeans.cluster_centers_):
        j = 0
        for d in d2vec.keys():
            y_kmeans2[j][i] = get_cos_similar(center, d2vec[d])
            j += 1
            
    j = 0
    for i in range(len(d_list)):
        if y_kmeans[i] != -1:
            y_kmeans[i] = y_kmeans2[j]
            j += 1
    return y_kmeans
            
def lm_grouping(intent_list, anno, q, q2d2d2score, d_list, d_json, cluster_num = 5, ):
    y_kmeans = np.zeros(len(d_list))
    d_set = set()
    for i in range(len(d_list)):
        d = d_list[i]
        if len(set(anno[q][str(d)]['anno']) & set(intent_list)) == 0:
            y_kmeans[i] = -1
        else:
            d_set = d_set | set(d_json[q][str(d)].keys())
    if len(d_set) < cluster_num:
        for i in range(len(d_list)):
            d = d_list[i]
            if len(set(anno[q][str(d)]['anno']) & set(intent_list)) == 0:
                pass
            else:
                y_kmeans[i] = 0
        return y_kmeans
    d_set = list(d_set)
    d_set_dic = dict([[d_set[idx],idx] for idx in range(len(d_set))])
    d2vec = {}
    for i in range(len(d_list)):
        if y_kmeans[i] == -1:
            continue
        d = d_list[i]
        d2vec[d] = np.zeros(len(d_set))
        for k,v in d_json[q][str(d)].items():
            d2vec[d][d_set_dic[str(k)]] = v

    vec_matrix = list(d2vec.values())
    kmeans = sklearn.cluster.KMeans(n_clusters = cluster_num, )
    y_kmeans2 = kmeans.fit_predict(vec_matrix)

    j = 0
    for i in range(len(d_list)):
        if y_kmeans[i] == -1:
            continue
        else:
            y_kmeans[i] = y_kmeans2[j]
            j += 1
    return y_kmeans

if __name__ == '__main__':
    # for q in q2d2d2score.keys():
    #     d_list = list(q2d2d2score[q].keys())
    #     grouping(q, q2d2d2score, d_list)
    # save roberta
    parser = argparse.ArgumentParser()
    parser.add_argument('-grouping_method',type=str, help= 'the method that group documents w.r.t. their sub-topic', required=False, default = 'gd')
    parser.add_argument('-m',type=int, help= 'the length of bert split trunks', required=False, default = 20)
    parser.add_argument('-kc',type=int, help= 'the number of trunks', required=False, default = 10)
    parser.add_argument('-kd',type=int, help= 'the number of bert split docs', required=False, default = 10)
    parser.add_argument('-multi_thread',type=str, required=False, default = 'True', choices=['True', 'False'])
    parser.add_argument('-click_rate',type=float, required=False, default = 0.9)
    parser.add_argument('-mode',type=str, required=False, default = 'exp_pre', choices=['exp', 'iter', 'sample', 'all', 'exp_all', 'exp_pre'])
    parser.add_argument('-click',type=str, required=False, default = 'no', choices=['no','add'])
    args = parser.parse_args()

    true_data = get_true_data(true_data, args, )
    tmp_anno = {}

    for i in range(len(true_data)):
        query_info = true_data[i]
        if type(query_info['intent']) == dict:
            total_intent = [int(intent) for intent in query_info['intent'].keys()]
        else:
            total_intent = [int(intent) for intent in query_info['intent']]
        q = true_data[i]['q']
        d_list = [true_data[i]['doc_list'][item]['did'] for item in range(len(true_data[i]['doc_list']))]
        intent_list = true_data[i]['intent']
        clustering_results = roberta_grouping(intent_list, anno, q, q2d2d2score, d_list=d_list, cluster_num = len(total_intent))
        true_data[i]['intent'] = [int(item) for item in list(set(clustering_results))]
        
        tmp_anno[q] = {}
        for idx, d in enumerate(d_list):
            tmp_anno[q][str(d)] = {'anno':[int(clustering_results[idx])]}
    json.dump(tmp_anno, open('tmp_anno.json', 'w'))


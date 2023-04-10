import json
import numpy as np
import tqdm
import argparse

def add2dic(dic1, u,q,ddd,para):
    if u not in dic1.keys():
        dic1[u] = {}
    if q not in dic1[u].keys():
        dic1[u][q] = {}
    dic1[u][q][ddd] = para

def para_search(u2info, user_list, file_name, mode):
    # input: a file_name of u2result_list including paras
    # return: u2q2ddd2para dict

    anno = json.load(open('../release/mode/anno.json'))
    para_dict = json.load(open(file_name))

    u2q2ddd2para = {}
    for u in user_list:
        idx = 0
        for raw_q in tqdm.tqdm(u2info[u]['raw_q2info'].keys()):
            q = u2info[u]['raw_q2info'][raw_q]['q']
            doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
            intent = u2info[u]['raw_q2info'][raw_q]['intent']
            if mode == 'synchronoize':
                for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys():
                    future_d_score = []
                    future_d_list = []
                    now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
                    for d in doc_list:
                        if d not in now_d_list:
                            future_d_list.append(d)               
                            future_d_score.append(int(intent) in anno[q][d]['anno'])
                    if len(future_d_score) < 2 or np.std(future_d_score) == 0:
                        continue
                    best_para = None
                    best_performance = 0
                    for paras in para_dict[u].keys():
                        para_arr = [float(item) for item in paras.split('_')]
                        # [idx]
                        performance = 0
                        for metric in ['map']:
                            performance += para_dict[u][paras]['bqe(bs+c)'][metric][idx]
                        if performance > best_performance:
                            best_performance = performance
                            best_para = paras
                    ddd = '_'.join(sorted(now_d_list))
                    add2dic(u2q2ddd2para, u, q, ddd, best_para) # dic1, u, q, ddd, para
                    idx += 1
            elif mode == 'asynchronize':
                raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
                for raw_d in [str(raw_d)]:
                    now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
                    now_d_score = []
                    now_d_score2_dic = {}
                    interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
                    d2score = {}
                    for d in interactions.keys():
                        goon = False
                        info_list = [0, 0]
                        for item in interactions[d]:
                            if item['motion'] == 'serp':
                                goon = True
                                if item['d'] not in now_d_score2_dic.keys():
                                    now_d_score2_dic[item['d']] = item['score']
                            elif item['motion'] == 'land':
                                now_d_score2_dic[item['d']] = item['score']
                                if goon:
                                    info_list[1] = 1
                                break
                        d2score[interactions[d][0]['d']] = info_list

                    for d in now_d_list:
                        if now_d_score2_dic[d] == 1 and d2score[d][1] == 1:
                            now_d_score.append(0)
                        else:
                            now_d_score.append(now_d_score2_dic[d])
                    now_d_score = [item - 1 for item in now_d_score]
                    if len(now_d_score) < 2 or np.std(now_d_score) == 0 or np.max(now_d_score) == 0:
                        continue

                    best_para = None
                    best_performance = 0
                    for paras in para_dict[u].keys():
                        performance = 0
                        for metric in ['ndcg@10']: # map
                            performance += para_dict[u][paras]['bqe(bs+c)'][metric][idx]
                        if performance > best_performance:
                            best_performance = performance
                            best_para = paras
                    ddd = '_'.join(sorted(now_d_list))
                    add2dic(u2q2ddd2para, u, q, ddd, best_para) # dic1, u, q, ddd, para
                    idx += 1
    return u2q2ddd2para

def get_best_performance(u2info, user_list, file_name, mode):
    anno = json.load(open('../release/mode/anno.json'))
    para_dict = json.load(open(file_name))

    u2q2ddd2para = {}
    for u in user_list:
        idx = 0
        for raw_q in tqdm.tqdm(u2info[u]['raw_q2info'].keys()):
            q = u2info[u]['raw_q2info'][raw_q]['q']
            doc_list = u2info[u]['raw_q2info'][raw_q]['doc_list']
            intent = u2info[u]['raw_q2info'][raw_q]['intent']
            if mode == 'synchronoize':
                for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys():
                    future_d_score = []
                    future_d_list = []
                    now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
                    for d in doc_list:
                        if d not in now_d_list:
                            future_d_list.append(d)               
                            future_d_score.append(int(intent) in anno[q][d]['anno'])
                    if len(future_d_score) < 2 or np.std(future_d_score) == 0:
                        continue
                    best_para = None
                    best_performance = 0
                    for paras in para_dict[u].keys():
                        # [idx]
                        performance = 0
                        for metric in ['map']:
                            performance += para_dict[u][paras]['bqe(bs+c)'][metric][idx]
                        if performance > best_performance:
                            best_performance = performance
                            best_para = paras
                    ddd = '_'.join(sorted(now_d_list))
                    add2dic(u2q2ddd2para, u, q, ddd, best_para) # dic1, u, q, ddd, para
                    idx += 1
            elif mode == 'asynchronize':
                raw_d = max([int(raw_d) for raw_d in u2info[u]['raw_q2task2info'][raw_q].keys()])
                for raw_d in [str(raw_d)]:
                    now_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['now_d_list']
                    now_d_score = []
                    now_d_score2_dic = {}
                    interactions = u2info[u]['raw_q2task2info'][raw_q][raw_d]['interactions']
                    d2score = {}
                    for d in interactions.keys():
                        goon = False
                        info_list = [0, 0]
                        for item in interactions[d]:
                            if item['motion'] == 'serp':
                                goon = True
                                if item['d'] not in now_d_score2_dic.keys():
                                    now_d_score2_dic[item['d']] = item['score']
                            elif item['motion'] == 'land':
                                now_d_score2_dic[item['d']] = item['score']
                                if goon:
                                    info_list[1] = 1
                                break
                        d2score[interactions[d][0]['d']] = info_list

                    for d in now_d_list:
                        if now_d_score2_dic[d] == 1 and d2score[d][1] == 1:
                            now_d_score.append(0)
                        else:
                            now_d_score.append(now_d_score2_dic[d])
                    now_d_score = [item - 1 for item in now_d_score]
                    if len(now_d_score) < 2 or np.std(now_d_score) == 0 or np.max(now_d_score) == 0:
                        continue

                    best_para = None
                    best_performance = 0
                    for paras in para_dict[u].keys():
                        para_arr = [float(item) for item in paras.split('_')]
                        # [idx]
                        performance = 0
                        for metric in ['ndcg@10']: # map
                            performance += para_dict[u][paras]['bqe(bs+c)'][metric][idx]
                        if performance > best_performance:
                            best_performance = performance
                            best_para = paras
                    ddd = '_'.join(sorted(now_d_list))
                    add2dic(u2q2ddd2para, u, q, ddd, best_para) # dic1, u, q, ddd, para
                    idx += 1
    return u2q2ddd2para

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',type=int, help= 'the user', required=False, default = 1000)
    args = parser.parse_args()

    u2info = json.load(open('../release/u2info.json'))
    user_list = [u for u in u2info.keys() if u.startswith('2_') == False]

    mode = 'udr' # udr rdr
    submode = ''
    u2q2ddd2para = para_search(u2info, user_list, file_name = f'../results/part3_{mode}/{mode}_para_search{submode}.json', mode=mode) 
    json.dump(u2q2ddd2para, open(f'../results/para/{mode}{submode}.json', 'w'))





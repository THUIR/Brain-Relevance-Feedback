import json
import numpy as np
seed = 2022

def search_by_file_path(u2results):
    metric = 'ndcg@10'
    para2performance = {}
    for u in u2results.keys():
        for para in u2results[u].keys():
            if para not in para2performance.keys():
                para2performance[para] = []
            para2performance[para] += (u2results[u][para]['bqe(bs+c-s)'][metric])

    best_para = []
    best_performance = 0

    select_arr = np.zeros(len(para2performance['0_0_0']))
    select_arr[:200] = 1
    np.random.shuffle(select_arr)
    select_arr = np.array(select_arr, dtype = bool)

    for para in para2performance.keys():
        performance = np.mean(np.array(para2performance[para])[select_arr])
        if performance > best_performance:
            best_para = [para]
            best_performance = performance
        elif performance == best_performance:
            best_para.append(para)

    print('best para', best_para, best_performance) 
    return best_para                                       

if __name__ == '__main__':
    np.random.seed(seed)
    base_path = '../results/part3_udr/udr_all.json'
    u2results = json.load(open(base_path))
    search_by_file_path(u2results)

    np.random.seed(seed)
    base_path = '../results/part3_rdr/rdr_all.json'
    u2results = json.load(open(base_path))
    search_by_file_path(u2results)
    








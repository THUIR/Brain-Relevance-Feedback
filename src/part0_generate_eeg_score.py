import os
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import random
from system.system_client import Eeg_classifier, Simulator_classifier
seed = 2021
random.seed(seed)
np.random.seed(seed)

def per_thread(u, u2info, reload, mode):
    if os.path.exists(f'../release/idx2eeg_score/{u}.json') and reload == False:
        return
    idx2eeg_score = {}
    y_true_list = {'land':[], 'serp':[]}
    y_pred_list = {'land':[], 'serp':[]}
    y_true_all = {'land':[], 'serp':[]}
    y_pred_all = {'land':[], 'serp':[], 'land_specific':[], 'land_general':[], 'serp_specific':[], 'serp_general':[]}
    re = {'land_all':[], 'serp':[], 'serp_all':[]}
    raw_q2raw_d2info = u2info[u]['raw_q2raw_d2info']
    classifier = Eeg_classifier(u)
    simple_classifier = Simulator_classifier()
    simple_classifier.shot = 500
    idx2eeg = json.load(open(f'../release/idx2eeg/{u}.json'))
    idx_sets = []
    for raw_q in raw_q2raw_d2info.keys():
        d2score = {}
        for item in raw_q2raw_d2info[raw_q]:
            idx = item['idx']
            if idx not in idx_sets:
                idx_sets.append(idx)
            else:
                continue    
            if item['motion'] in ['serp']:   
                d2score[item['raw_d']] = item['score']
            eeg = idx2eeg[str(idx)]
            anno = 1 if item['score'] > 1 and item['motion'] in ['serp'] or item['score'] > 2 and item['motion'] in ['land'] else 0
            if len(eeg['fs']) > 0:
                y_pred, y_pred_general, y_pred_specific = classifier.add_data(raw_q, item['raw_d'], eeg, anno, item['motion'] in ['land'])
            else:
                tmp_mean = np.mean(classifier.y_pred) if len(classifier.y_pred) > 20 else 0.4482
                simple_classifier.para['anno'][0]['loc'] = tmp_mean - 0.04
                simple_classifier.para['anno'][1]['loc'] = tmp_mean + 0.04
                y_pred = simple_classifier.add_data(raw_q, item['raw_d'], eeg, anno, item['motion'] in ['land'])
                y_pred_specific = y_pred
                y_pred_general = y_pred
            idx2eeg_score[idx] = y_pred
            y_pred_all[f"{item['motion']}_specific"].append(y_pred_specific)
            y_pred_all[f"{item['motion']}_general"].append(y_pred_general)
            y_true_list[item['motion']].append(anno)
            y_pred_list[item['motion']].append(y_pred)
        classifier.add_label(raw_q, d2score)
        if len(y_true_list['serp']) > 0:
            if np.std(y_true_list['serp']) > 0:
                re['serp'].append(roc_auc_score(y_true_list['serp'], y_pred_list['serp']))
            y_true_all['serp'] += y_true_list['serp']
            y_pred_all['serp'] += y_pred_list['serp']
            y_true_list['serp'] = []
            y_pred_list['serp'] = []
    y_true_all['land'] = y_true_list['land']
    y_pred_all['land'] = y_pred_list['land']
    re['land_all'] = roc_auc_score(y_true_all['land'], y_pred_all['land'])
    re['serp_all'] = roc_auc_score(y_true_all['serp'], y_pred_all['serp'])
    print(u, 'serp per task:', np.mean(re['serp']), 'serp all:', roc_auc_score(y_true_all['serp'], y_pred_all['serp']), 'landing page all:', roc_auc_score(y_true_all['land'], y_pred_all['land']) if len(y_true_all['land']) > 0 else '-')

    json.dump(idx2eeg_score, open(f'../release/idx2eeg_score/{u}.json', 'w'))
    re['serp_info'] = [y_true_all['serp'], y_pred_all['serp'], y_pred_all['serp_specific'], y_pred_all['serp_general']]
    re['land_info'] = [y_true_all['land'], y_pred_all['land'], y_pred_all['land_specific'], y_pred_all['land_general']]
    return re

def print_error(e):
    print('e: ', e)

def load_eeg2(u2info, reload= False, mode = ''):
    u_list = list(u2info.keys())
    from multiprocessing.pool import Pool
    pool = Pool(40)
    task_list = []
    all_metric_dic = {}
    for u in u_list:
        task_list.append(pool.apply_async(per_thread, args = (u, u2info, reload, mode), error_callback = print_error))
    pool.close()
    pool.join()
    for i, task in enumerate(task_list):
        all_metric_dic[u_list[i]] = task.get()
    for key in ['serp', 'serp_all', 'land_all']:
        print(key, np.mean(list([np.mean(value[key]) for value in all_metric_dic.values()])))
    json.dump(all_metric_dic, open(f'../results/part0_eeg_prediction_performance.json', 'w'))

if __name__ == '__main__':
    u2info = json.load(open('../release/u2info.json'))
    load_eeg2(u2info, True, )
    











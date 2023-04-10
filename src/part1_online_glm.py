import json
import numpy as np

u2info = json.load(open('../release/u2info.json'))
anno = json.load(open('../release/mode/anno.json'))

total_dic = {'method':[],'u':[],'raw_d':[],'task':[],'anno':[],'score':[]}

for u in u2info.keys():
    if u.startswith('2_'):
        for raw_q in u2info[u]['raw_q2info'].keys():
            if 'method_list' not in u2info[u]['raw_q2info'][raw_q].keys():
                continue
            method_list = u2info[u]['raw_q2info'][raw_q]['method_list']
            # 最后一个raw_d实际上是用户没有接触的，所以不考虑
            raw_d_list = [int(item) for item in list(u2info[u]['raw_q2task2info'][raw_q].keys())]

            if len(method_list) != len(raw_d_list) - 1:
                # jiayudebug snippet start----------
                inputs = ''
                while inputs != 'continue':
                    try:
                        print(eval(inputs))
                    except Exception as e:
                        print(e)
                    inputs = input()
                # jiayudebug snippet end-------------
                    
            intent = u2info[u]['raw_q2info'][raw_q]['intent']
            q = u2info[u]['raw_q2info'][raw_q]['q']
            task_effect = np.sum([int(intent) in anno[q][d]['anno'] for d in u2info[u]['raw_q2info'][raw_q]['doc_list']]) / len(u2info[u]['raw_q2info'][raw_q]['doc_list'])
            for i, raw_d_int in enumerate(sorted(raw_d_list)[:-1]):
                raw_d = str(raw_d_int)
                method = 1 if method_list[i]['method_choosen'] == 'bert-bs+click' else 0
                future_d_list = u2info[u]['raw_q2task2info'][raw_q][raw_d]['future_d_list']
                future_d_score = u2info[u]['raw_q2task2info'][raw_q][raw_d]['future_d_score']
                total_dic['task'].append(task_effect)
                total_dic['raw_d'].append(int(raw_d))
                total_dic['method'].append(method)
                total_dic['u'].append(u)
                total_dic['anno'].append(1 if int(intent) in anno[q][future_d_list[0]]['anno'] else 0)
                total_dic['score'].append(future_d_score[0])

json.dump(total_dic, open('../results/part1_glm_dict.json', 'w'))

import statsmodels.formula.api as smf

md = smf.mixedlm(f"anno ~ method + task + C(u) + raw_d", total_dic, re_formula="~method", groups="u")
mdf = md.fit()
print('mixed linear anno: ', mdf.summary())
    

md = smf.mixedlm(f"score ~ method + task + C(u) + raw_d", total_dic, re_formula="~method", groups="u")
mdf = md.fit()
print('mixed linear score: ',mdf.summary())



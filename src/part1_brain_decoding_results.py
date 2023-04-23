import json
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

def evaluate(eeg_performance, dis_select = '2'):
    metric = {'combined':[],'presonalized':[],'generalized':[],'serp':[], 'land':[], 'combined_cold_start':[], 'presonalized_cold_start':[],'generalized_cold_start':[],'presonalized_serp':[],'presonalized_land':[],'generalized_serp':[],'generalized_land':[]}
    for u in eeg_performance.keys():
        if type(dis_select) == list:
            flag = False
            for item in dis_select:
                flag = u.startswith(item) or flag
        else:
            flag = u.startswith(dis_select)
        if  flag == False:
            y_pred = eeg_performance[u]['serp_info'][1] + eeg_performance[u]['land_info'][1]
            y_true = eeg_performance[u]['serp_info'][0] + eeg_performance[u]['land_info'][0]
            metric['combined'].append(roc_auc_score(y_true, y_pred))
            metric['combined_cold_start'].append(roc_auc_score(y_true[:100], y_pred[:100]))

            y_pred = eeg_performance[u]['serp_info'][2] + eeg_performance[u]['land_info'][2]
            y_pred = [np.mean(y_pred) for i in range(40)] + y_pred[40:]
            metric['presonalized'].append(roc_auc_score(y_true, y_pred))
            metric['presonalized_cold_start'].append(roc_auc_score(y_true[:100], y_pred[:100]))

            y_pred = eeg_performance[u]['serp_info'][3] + eeg_performance[u]['land_info'][3]
            metric['generalized'].append(roc_auc_score(y_true, y_pred))
            metric['generalized_cold_start'].append(roc_auc_score(y_true[:100], y_pred[:100]))

            y_pred = eeg_performance[u]['serp_info'][1]
            y_true = eeg_performance[u]['serp_info'][0]
            metric['serp'].append(roc_auc_score(y_true, y_pred))
            
            y_pred = eeg_performance[u]['land_info'][1]
            y_true = eeg_performance[u]['land_info'][0]
            metric['land'].append(roc_auc_score(y_true, y_pred))    

            y_pred = eeg_performance[u]['serp_info'][2]
            y_true = eeg_performance[u]['serp_info'][0]
            metric['presonalized_serp'].append(roc_auc_score(y_true, y_pred))

            y_pred = eeg_performance[u]['land_info'][2]
            y_true = eeg_performance[u]['land_info'][0]
            metric['presonalized_land'].append(roc_auc_score(y_true, y_pred))
            
            y_pred = eeg_performance[u]['serp_info'][3]
            y_true = eeg_performance[u]['serp_info'][0]
            metric['generalized_serp'].append(roc_auc_score(y_true, y_pred))

            y_pred = eeg_performance[u]['land_info'][3]
            y_true = eeg_performance[u]['land_info'][0]
            metric['generalized_land'].append(roc_auc_score(y_true, y_pred)) 
            
            if u.startswith('2_'):
                print(u, metric['combined'][-1])
               
    return metric

def print_result(dis_select = '2'):
    eeg_performance = json.load(open(f'../results/part0_eeg_prediction_performance.json'))
    metric = evaluate(eeg_performance, dis_select)    

    for key in metric.keys():
        print(key, np.mean(metric[key]), np.std(metric[key]))
    for testing_pair in [['combined', 'presonalized'],['combined', 'generalized'],['combined_cold_start', 'presonalized_cold_start'],['combined_cold_start', 'generalized_cold_start'], ['generalized_serp', 'presonalized_serp'], ['generalized_land', 'presonalized_land'], ]:
        p=ttest_rel(metric[testing_pair[0]], metric[testing_pair[1]]).pvalue
        print(f'{testing_pair[0]} vs {testing_pair[1]}: p={p}')


def paint_result():
    def ax2_init(ax2, length, dic):
        ax2.set_ylim(0.4, 0.9)
        y_sticks = np.arange(0.4,0.9,0.05)
        ax2.set_yticks(y_sticks)
        ax2.set_yticklabels([str("%.2f"%round((int(item*100+0.1))/100, 2)) for item in y_sticks],fontsize=13)
        ax2.xaxis.tick_bottom()

        ax2.set_xlim([-0.5, length - 1 + .5])
        x_sticks = np.arange(0,length - 1 +.1,1)
        ax2.set_xticks(x_sticks)
        ax2.set_xticklabels([dic[int(x_sticks[i])] for i in range(len(x_sticks))], fontsize=13)
        ax2.plot()

        for i in range(length + 1):
            ax2.vlines(i, 0, 1, linestyles='dotted')

    def paint(a,label,m, ax2, dic=None):
        if dic == None:
            tmp = [[i, a[i]] for i in range(len(a))]
            tmp = sorted(tmp, key = lambda item: item[1], reverse = True)
            dic = {}
            for i in range(len(a)):
                dic[i] = tmp[i][0]
        ax2.scatter([i for i in range(len(a))], [a[dic[i]] for i in range(len(a))], label=label, s=60, marker=m)
        return dic

    fig, ax2 = plt.subplots(figsize=(10,5))
    eeg_performance = json.load(open(f'../results/part0_eeg_prediction_performance.json'))
    metric = evaluate(eeg_performance)  

    dic1 = paint(metric['serp'], 'Snippet', 'o', ax2, )
    paint(metric['land'], 'Landing page','*',ax2, dic1)

    ax2_init(ax2, length = len([key for key in eeg_performance.keys() if key.startswith('2')==False]), dic=dic1)


    plt.xlabel("Participant ID",fontsize=19)
    plt.ylabel("AUC",fontsize=19)

    plt.hlines(0.5, -0.5, 40, linestyles='solid', colors='red', label='baseline')
    plt.legend(fontsize=15,loc="upper right")

    plt.savefig('../results/brain_decoding_serp_vs_land.jpg', bbox_inches='tight')

    plt.cla()

    # 画不同method的平均表现
    
    dic1 = paint(metric['combined'], 'F(x)', 'o',ax2, )
    paint(metric['presonalized'], '$F_p(x)$','*',ax2, dic1)
    paint(metric['generalized'], '$F_g(x)$','v',ax2, dic1)

    ax2_init(ax2, length = len([key for key in eeg_performance.keys() if key.startswith('2')==False]), dic=dic1)

    plt.xlabel("Participant ID",fontsize=19)
    plt.ylabel("AUC",fontsize=19)

    plt.hlines(0.5, -0.5, 40, linestyles='solid', colors='red', label='baseline')
    plt.legend(fontsize=15,loc="upper right")

    plt.savefig('../results/brain_decoding_p_vs_g.jpg', bbox_inches='tight')

def print_non_click_result():
    u2info = json.load(open('../release/u2info.json'))
    u2idx2score = {}
    for u in u2info.keys():
        if u.startswith('2') == False:
            u2idx2score[u] = json.load(open(f'../release/idx2eeg_score/{u}.json'))

    u2mean_score = {}
    u2std_score = {}

    for u in u2idx2score.keys():
        u2mean_score[u] = np.mean(list(u2idx2score[u].values()))
        u2std_score[u] = np.std(list(u2idx2score[u].values()))


    from sklearn.metrics import roc_auc_score
    u2auc = {}

    for u in u2info.keys():
        if u.startswith('2') == False:
            y_pred = {'non-click':[],'bad-click':[]}
            y_true = {'bad-click':[],'non-click':[]}
            idx_sets = []
            for raw_q in u2info[u]['raw_q2raw_d2info'].keys():
                d2score = {}
                goon = False
                idx_len = 0
                for item in u2info[u]['raw_q2raw_d2info'][raw_q]:
                    d = item['d']
                    if d not in d2score.keys():
                        info_list = [0, 0, 0, 0, 0]
                    else:
                        info_list = d2score[d]
                    if item['motion'] == 'land':
                        info_list[3] = item['score']
                        if str(item['idx']) in u2idx2score[u].keys():
                            info_list[4] = u2idx2score[u][str(item['idx'])] 
                        if goon:
                            info_list[1] = 1
                    else:
                        goon = True
                    d2score[d] = info_list
                for item in u2info[u]['raw_q2raw_d2info'][raw_q]:
                    idx = item['idx']
                    if idx not in idx_sets:
                        idx_sets.append(idx)
                    else:
                        continue 
                    d = item['d']
                    if d not in d2score.keys():
                        info_list = [0, 0, 0, 0, 0]
                    else:
                        info_list = d2score[d]
                    if item['motion'] == 'serp':
                        idx_len += 1
                        goon = True
                        if str(item['idx']) in u2idx2score[u].keys():
                            info_list[0] = u2idx2score[u][str(item['idx'])] 
                        info_list[2] = item['score']
                    d2score[d] = info_list
                    if item['motion'] == 'serp' and info_list[1] == 0:
                        y_pred['non-click'] += [info_list[0]]
                        y_true['non-click'] += [1 if info_list[2] > 1 else 0]
            u2auc[u] = roc_auc_score(y_true['non-click'], y_pred['non-click'])
    print('non-click auc: ', np.mean(list(u2auc.values())))

if __name__ == '__main__':
    # print brain decoding results
    print_result(dis_select = ['0','1'])
    # paint brain decoding results into images
    paint_result()
    # print brain decoding results for non-click data samples
    print_non_click_result()
    

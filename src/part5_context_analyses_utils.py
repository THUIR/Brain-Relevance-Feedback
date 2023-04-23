import json
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

def append2dic(dic1,para,performance):
    if para not in dic1.keys():
        dic1[para] = []
    dic1[para].append(performance)

def udr_click():
    u2result_list = json.load(open('../results/part3_udr/udr_all.json'))
    baseline = json.load(open('../results/part3_udr/udr_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {0:{},1:{},2:{}}

    for u in u2result_list.keys():
        for para_str in u2result_list[u].keys():
            para = [int(item) for item in para_str.split('_')]
            if para not in [[1,1,0],[1,1,3],[1,0,3],[1,0,0], [1,1,5], [1,5,1]]:
                continue
            if para == [1,1,3]:
                rate = 'best'
            elif para == [1,1,0]:
                rate = 'ablation_no_bs'
            elif para == [1,0,0]:
                rate = 'baseline'
            elif para == [1,0,3]:
                rate = 'ablation_no_c'
            elif para == [1,1,5]:
                rate = 'bs'
            elif para == [1,5,1]:
                rate = 'c'
            for i in range(len(u2result_list[u][para_str][method]['information'])):
                c = min(u2result_list[u][para_str][method]['information'][i]['c'], 2)
                performance = u2result_list[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    method = 'bert'
    for u in baseline.keys():
        for para_str in ['0_0_0']:
            rate = 'baseline'
            for i in range(len(baseline[u][para_str][method]['information'])):
                c = min(baseline[u][para_str][method]['information'][i]['c'], 2)
                performance = baseline[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)
    return c2para2performance

def paint_performance_regarding_context_udr():
    # get data
    c2para2performance = udr_click()

    total_width, n = 0.6, 3
    width = total_width / n
    fontsize=30
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharey=True)
    plt.xlabel('#Clicks', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    plt.ylim(0.15,0.5)
    colors = ['#e38c7a', '#dccfcc', '#bbc2d4']
    plt.xticks([0,1,2],['$0~(51.5\%)$','$1~(20.9\%)$','$\geq 2~(19.3\%)$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    c2str = {
        'c': r"$\theta^{u,p:c:bs}$=1:5:1",
        'bs': r"$\theta^{u,p:c:bs}$=1:1:5",
        'best': r"$\theta^{u,p:c:bs}$=1:1:3",
    }
    from textwrap import wrap

    for k,para in enumerate(['c','bs','best']): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
        plt.bar([i - total_width / 3 + width * k for i in range(len(y))], y, width=width, label = c2str[para], color = colors[k], alpha=0.75)
    
    labels = [ '\n'.join(wrap(l, 19)) for l in c2str.values()]
    plt.legend(labels, fontsize=25, handlelength=1)
    plt.savefig('../results/parameter_udr.png',bbox_inches='tight')
    plt.show()

def rdr_bad_click():
    u2result_list = json.load(open('../results/part3_rdr/rdr_all.json'))
    baseline = json.load(open('../results/part3_rdr/rdr_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {0:{},1:{},2:{}}
    # 分析多个曲线：强调bs 1:1:5,强调c 1:5:1 强调p 5:1:1

    for u in u2result_list.keys():
        for para_str in u2result_list[u].keys():
            para = [int(item) for item in para_str.split('_')]
            if para not in [[0,2,0], [0,2,5], [0,5,1], [0,1,5]]:
                continue
            if para == [0,2,5]:
                rate = 'best'
            elif para == [0,5,1]:
                rate = 'c'
            elif para == [0,1,5]:
                rate = 'bs'
            elif para == [0,2,0]:
                rate = 'ablation_no_bs'
            for i in range(len(u2result_list[u][para_str][method]['information'])):
                c = min(u2result_list[u][para_str][method]['information'][i]['c'], 2)
                c = max(c, 0)
                performance = u2result_list[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    method = 'bert'
    for u in baseline.keys():
        for para_str in ['0_0_0']:
            rate = 'baseline'
            for i in range(len(baseline[u][para_str][method]['information'])):
                c = min(baseline[u][para_str][method]['information'][i]['c'], 2)
                c = max(c, 0)
                performance = baseline[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)
    return c2para2performance

def udr_length():
    u2result_list = json.load(open('../results/part3_udr/udr_all.json'))
    baseline = json.load(open('../results/part3_udr/udr_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{}}
    # analyze various of combination parameters, i.e., parameters with higher weight for brain signals: bs; parameters with higher weight for pseudo-relevance signals: p; parameters with higher weight for click signals: c; 

    for u in u2result_list.keys():
        for para_str in u2result_list[u].keys():
            para = [int(item) for item in para_str.split('_')]
            if para not in [[1,1,0],[1,1,3],[1,0,3],[1,0,0]]:
                continue
            if para == [1,1,3]:
                rate = 'best'
            elif para == [1,1,0]:
                rate = 'ablation_no_bs'
            elif para == [1,0,0]:
                rate = 'baseline'
            else:
                rate = 'ablation_no_c'
            for i in range(len(u2result_list[u][para_str][method]['information'])):
                c = min(u2result_list[u][para_str][method]['information'][i]['d_len'], 10)
                performance = u2result_list[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    method = 'bert'
    for u in baseline.keys():
        for para_str in ['0_0_0']:
            rate = 'baseline'
            for i in range(len(baseline[u][para_str][method]['information'])):
                c = min(baseline[u][para_str][method]['information'][i]['d_len'], 10)
                performance = baseline[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)
   
    for k,para in enumerate(['ablation_no_bs','best']): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]) - np.mean(c2para2performance[c]['baseline']))
            x.append(str(c))
    return c2para2performance

def rdr_click():
    u2result_list = json.load(open('../results/part3_rdr/rdr_all.json'))
    baseline = json.load(open('../results/part3_rdr/rdr_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {0:{}, 1:{},2:{},3:{},4:{},5:{},}

    for u in u2result_list.keys():
        for para_str in u2result_list[u].keys():
            para = [int(item) for item in para_str.split('_')]
            if para not in [[0,2,0],[0,2,5],[0,1,5]]:
                continue
            if para == [0,2,5]:
                rate = 'best'
            elif para == [0,2,0]:
                rate = 'ablation_no_bs'
            elif para == [0,1,5]:
                rate = 'bs'
            
            for i in range(len(u2result_list[u][para_str][method]['information'])):
                c = min(u2result_list[u][para_str][method]['information'][i]['click'], 5)
                c = max(c, 0)
                performance = u2result_list[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    method = 'bert'
    for u in baseline.keys():
        for para_str in ['0_0_0']:
            rate = 'baseline'
            for i in range(len(baseline[u][para_str][method]['information'])):
                c = min(baseline[u][para_str][method]['information'][i]['click'], 5)
                c = max(c, 1)
                performance = baseline[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    for k,para in enumerate(['ablation_no_bs','best']): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
    return c2para2performance

def rdr_length():
    u2result_list = json.load(open('../results/part3_rdr/rdr_all.json'))
    baseline = json.load(open('../results/part3_rdr/rdr_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {1:{},2:{},3:{},4:{}}

    for u in u2result_list.keys():
        for para_str in u2result_list[u].keys():
            para = [int(item) for item in para_str.split('_')]
            if para not in [[0,2,0],[0,2,5],[0,1,5]]:
                continue
            if para == [0,2,5]:
                rate = 'best'
            elif para == [0,2,0]:
                rate = 'ablation_no_bs'
            elif para == [0,1,5]:
                rate = 'bs'
            
            for i in range(len(u2result_list[u][para_str][method]['information'])):
                c = int((u2result_list[u][para_str][method]['information'][i]['d_len'] - 1) / 5) + 1
                c = min(c, 4)
                c = max(c, 1)
                performance = u2result_list[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    method = 'bert'
    for u in baseline.keys():
        for para_str in ['0_0_0']:
            rate = 'baseline'
            for i in range(len(baseline[u][para_str][method]['information'])):
                c = int((baseline[u][para_str][method]['information'][i]['d_len'] - 1) / 5) + 1
                c = min(c, 4)
                c = max(c, 1)
                performance = baseline[u][para_str][method]['ndcg@10'][i]
                append2dic(c2para2performance[c], rate, performance)

    for k,para in enumerate(['ablation_no_bs','best']): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
    return c2para2performance

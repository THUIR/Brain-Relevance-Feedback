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
    u2result_list = json.load(open('../results/part3_synchronoize/syn_all.json'))
    baseline = json.load(open('../results/part3_synchronoize/syn_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {0:{},1:{},2:{}}
    # 分析多个曲线：强调bs 1:1:5,强调c 1:5:1 强调p 5:1:1

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
    u2result_list = json.load(open('../results/part3_asynchronize/asy_all.json'))
    baseline = json.load(open('../results/part3_asynchronize/asy_baseline.json')) 
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

def paint_performance_regarding_context_rdr():
    c2para2performance = rdr_bad_click()
    
    total_width, n = 0.6, 3
    width = total_width / n
    fontsize=30
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharey=True)
    plt.xlabel('#Bad Clicks', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    plt.ylim(0.2,0.9)
    colors = ['#e38c7a', '#dccfcc', '#bbc2d4']
    plt.xticks([0,1,2],['$0~(64.8\%)$','$1~(27.4\%)$','$\geq 2~(7.8\%)$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    c2str = {
        'c': r"$\theta^{r,p:c:bs}$=0:5:1",
        'bs': r"$\theta^{r,p:c:bs}$=0:1:5",
        'best': r"$\theta^{r,p:c:bs}$=0:2:5",
    }

    # from textwrap import wrap
    for k,para in enumerate(['c','bs','best']): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
            print(len(c2para2performance[c][para]), y)
        plt.bar([i - total_width / 3 + width * k for i in range(len(y))], y, width=width, label = c2str[para], color = colors[k], alpha=0.75)

    plt.legend(fontsize=26, handlelength=1)
    plt.savefig('../results/parameter_rdr.png',bbox_inches='tight')
    plt.show()

def paint_performance_regarding_context_udr_fixed_parameters2():
    c2para2performance = udr_length()
    # from textwrap import wrap
    fontsize=43
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
    plt.xlabel('Historical documents($h$)', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    colors = ['#e38c7a', '#dccfcc']
    c2str = {'best': "$QE^{R^{bs}\oplus R^c\oplus R^p}$",
            'ablation_no_bs': "$QE^{R^c\oplus R^p}$"
            }
    plt.xticks([0,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,'$\geq 9$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    para2y = {}
    
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        plt.plot([i - 1 for i in range(len(y))][1:], y[1:], label = c2str[para], lw = 4, color = colors[k], alpha=0.75)

    for i in range(1, len(y)):
        y_max = max(para2y['best'][i], para2y['ablation_no_bs'][i])
        y_min = min(para2y['best'][i], para2y['ablation_no_bs'][i])
        plt.vlines(i - 1, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)

    for k,c in enumerate(c2para2performance.keys()):
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            y_median = (para2y['best'][k] + para2y['ablation_no_bs'][k]) / 2
            plt.scatter([k-1], [y_median], marker = '*', c='red', lw=2)
        
    plt.legend(fontsize=fontsize, handlelength=1)
    plt.savefig('../results/length_udr.png',bbox_inches='tight')

def udr_length():
    u2result_list = json.load(open('../results/part3_synchronoize/syn_all.json'))
    baseline = json.load(open('../results/part3_synchronoize/syn_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {1:{},2:{},3:{},4:{},5:{},6:{},7:{},8:{},9:{},10:{}}
    # 分析多个曲线：强调bs 1:1:5,强调c 1:5:1 强调p 5:1:1

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

def paint_performance_regarding_context_udr_fixed_parameters1():
    c2para2performance = udr_click()

    fontsize=43
    fig = plt.figure(figsize=(15, 10)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[25,20]) 
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    colors = ['#e38c7a', '#dccfcc']
    c2str = {'best': "$QE^{R^{bs}\oplus R^c\oplus R^p}$",
            'ablation_no_bs': "$QE^{R^c\oplus R^p}$"
            }
    para2y = {}
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]) )
            x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        ax1.plot([i for i in range(len(y))], y, label = c2str[para], lw = 4, color = colors[k], alpha=0.75)
        ax2.plot([i for i in range(len(y))], y, label = c2str[para], lw = 4, color = colors[k], alpha=0.75)
    for i in range(0, len(y)):
        y_max = max(para2y['best'][i], para2y['ablation_no_bs'][i])
        y_min = min(para2y['best'][i], para2y['ablation_no_bs'][i])
        ax1.vlines(i, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)
        ax2.vlines(i, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)

    for k,c in enumerate(c2para2performance.keys()):
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            y_median = (para2y['best'][k] + para2y['ablation_no_bs'][k]) / 2
            ax1.scatter([c], [y_median], marker = '*', c='red', lw=2)
            ax2.scatter([c], [y_median], marker = '*', c='red', lw=2)

    ax1.set_ylim(.45, .475)     
    ax2.set_ylim(0.29,0.31)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
            
    ax1.legend(fontsize=fontsize, handlelength=1)
    plt.xlabel('#Click', fontsize=fontsize)
    ax2.set_xticks([0,1,2,],['0~(51.5\%)','$1~(20.9\%)$','$\geq 2~(19.3\%)$'], fontsize=fontsize)
    ax1.set_yticks([0.45,0.475],[0.45,0.475], fontsize=fontsize)
    ax2.set_yticks([0.29,0.30,0.31],[0.275,0.30,0.31], fontsize=fontsize)
    ax1.set_xticks([])
    fig.tight_layout(pad=0.8)
    ax1.set_xlim(-0.5,2.5)
    ax2.set_xlim(-0.5,2.5)
    # ax1.set_ylabel('NDCG@10', fontsize=fontsize, loc=)
    fig.text(-0.01, 0.5, 'NDCG@10', va='center', rotation='vertical', fontsize = fontsize)
    plt.savefig('../results/udr1.png',bbox_inches='tight')

def rdr_click():
    u2result_list = json.load(open('../results/part3_asynchronize/asy_all.json'))
    baseline = json.load(open('../results/part3_asynchronize/asy_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {0:{}, 1:{},2:{},3:{},4:{},5:{},}
    # 分析多个曲线：强调bs 1:1:5,强调c 1:5:1 强调p 5:1:1

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
    u2result_list = json.load(open('../results/part3_asynchronize/asy_all.json'))
    baseline = json.load(open('../results/part3_asynchronize/asy_baseline.json')) 
    method = 'bqe(bs+c-s)'

    c2para2performance = {1:{},2:{},3:{},4:{}}
    # 分析多个曲线：强调bs 1:1:5,强调c 1:5:1 强调p 5:1:1

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

def paint_performance_regarding_context_rdr_fixed_parameters3():
    c2para2performance = rdr_length()
    fontsize=45
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(12, 10), sharey=True)
    plt.xlabel('Session Length~($h_{max}$)', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    colors = ['#e38c7a', '#dccfcc']
    plt.xticks([0,1,2,3,],['5', '10','15', '20'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    para2y = {}
    c2str = {'best': "$R^{bs}\oplus R^c\oplus R^p$",
            'ablation_no_bs': "$R^c\oplus R^p$"
            }
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
            print(len(c2para2performance[c][para]))
        para2y[para] = y
        plt.plot([i for i in range(len(y))], y, label = c2str[para], lw = 4, color = colors[k], alpha=0.75)

    for i in range(0, len(y)):
        y_max = max(para2y['best'][i], para2y['ablation_no_bs'][i])
        y_min = min(para2y['best'][i], para2y['ablation_no_bs'][i])
        plt.vlines(i, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)

    for k,c in enumerate(c2para2performance.keys()):
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            y_median = (para2y['best'][k] + para2y['ablation_no_bs'][k]) / 2
            plt.scatter([k], [y_median], marker = '*', c='red', lw=2)
        
    plt.legend(fontsize=fontsize, handlelength=1)
    plt.savefig('../results/rdr3.png',bbox_inches='tight')

def paint_performance_regarding_context_rdr_fixed_parameters1():
    c2para2performance = rdr_bad_click()
    fontsize=45
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(12, 10), sharey=True)
    plt.xlabel('#Bad Click', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    colors = ['#e38c7a', '#dccfcc']
    plt.xticks([0,1,2,],['$0~(64.8\%)$', '$1~(27.4\%)$','$\geq 2~(7.8\%)$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    para2y = {}
    c2str = {'best': "$R^{bs}\oplus R^c\oplus R^p$",
            'ablation_no_bs': "$R^c\oplus R^p$"
            }
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        plt.plot([i for i in range(len(y))], y, label = c2str[para], lw = 4, color = colors[k], alpha=0.75)

    for i in range(0, len(y)):
        y_max = max(para2y['best'][i], para2y['ablation_no_bs'][i])
        y_min = min(para2y['best'][i], para2y['ablation_no_bs'][i])
        plt.vlines(i, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)

    for k,c in enumerate(c2para2performance.keys()):
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            y_median = (para2y['best'][k] + para2y['ablation_no_bs'][k]) / 2
            plt.scatter([k], [y_median], marker = '*', c='red', lw=2)
        
    plt.legend(fontsize=fontsize, handlelength=1)
    plt.savefig('../results/rdr1.png',bbox_inches='tight')

def paint_performance_regarding_context_rdr_fixed_parameters2():
    c2para2performance = rdr_click()
    fontsize=45
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(12, 10), sharey=True)
    plt.xlabel('#Click', fontsize=fontsize)
    plt.ylabel('NDCG@10', fontsize=fontsize)
    # plt.ylim(0.2,0.9)
    colors = ['#e38c7a', '#dccfcc']
    plt.xticks([0,1,2,3,4,],[1,2,3,4,'$\geq 5$'],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    para2y = {}
    c2str = {'best': "$R^{bs}\oplus R^c\oplus R^p$",
            'ablation_no_bs': "$R^c\oplus R^p$"
            }
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            if c > 0:
                y.append(np.mean(c2para2performance[c][para]))
                x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        plt.plot([i for i in range(len(y))], y, label = c2str[para], lw = 4, color = colors[k], alpha=0.75)

    for i in range(0, len(y)):
        y_max = max(para2y['best'][i], para2y['ablation_no_bs'][i])
        y_min = min(para2y['best'][i], para2y['ablation_no_bs'][i])
        plt.vlines(i, ymin = y_min , ymax = y_max, color='black', lw=2, linestyle = 'dashed',zorder=-1)

    for k,c in enumerate(c2para2performance.keys()):
        if c > 0 and stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            y_median = (para2y['best'][k-1] + para2y['ablation_no_bs'][k-1]) / 2
            plt.scatter([k-1], [y_median], marker = '*', c='red', lw=2)
        
    plt.legend(fontsize=fontsize, handlelength=1)
    plt.savefig('../results/length_rdr.png',bbox_inches='tight')

if __name__ == '__main__':

    # print images for performance comparaison in different scenarios with different parameters
    # paint_performance_regarding_context_udr()
    # paint_performance_regarding_context_rdr()

    # print images for performance comparaison in different scenarios
    # paint_performance_regarding_context_udr_fixed_parameters1()
    # paint_performance_regarding_context_udr_fixed_parameters2()
    # paint_performance_regarding_context_rdr_fixed_parameters1()
    # paint_performance_regarding_context_rdr_fixed_parameters2()
    paint_performance_regarding_context_rdr_fixed_parameters3()

from part5_context_analyses_utils import udr_click, rdr_bad_click
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import matplotlib
import seaborn as sns

def paint_performance_regarding_context_udr():
    # get data
    c2para2performance = udr_click()

    total_width, n = 0.8, 3
    width = total_width / n
    fontsize=26
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharey=True)
    ax2 = ax1.twinx()
    plt.grid(alpha=0.5,axis='y')
    ax1.set_xlabel('#Click', fontsize=fontsize)
    ax1.set_ylabel('NDCG@10', fontsize=fontsize)
    ax2.set_ylabel("IMP. (%)",fontsize=fontsize,color='blue',)
    ax1.set_ylim(0.15,0.55)
    colors = ['#9BA3BA','#e38c7a', '#C89598']
    ax1.set_xticks([0,1,2],['$0~(51.5\%)$','$1~(20.9\%)$','$\geq 2~(19.3\%)$'],fontsize=fontsize)
    ax1.set_yticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=fontsize,rotation=30)
    c2str = {
        'bs': r"$\theta^{it,bs:c:p}$=5:1:1",
        'best': r"$\theta^{it,bs:c:p}$=3:1:1",
        'c': r"$\theta^{it,bs:c:p}$=1:5:1",
    }
    from textwrap import wrap

    for k,para in enumerate(['bs','best','c',]): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
        ax1.bar([i - total_width / 3 + width * k for i in range(len(y))],  y, linewidth = 2, edgecolor = 'black', width=width-0.02, label = c2str[para], color = colors[k], alpha=0.5,)

    for com_paris in [['best','bs',"5:1:1 vs 1:1:3",'dashed',2],['best','c',"1:5:1 vs 1:1:3",'solid',0],]:
        y = []
        x = []
        for k,c in enumerate(c2para2performance.keys()):
            y.append(((np.mean(c2para2performance[c][com_paris[1]])-np.mean(c2para2performance[c][com_paris[0]]))/np.mean(c2para2performance[c][com_paris[0]])+0.05)*1+0.2)
            x.append(str(c))
            if stats.ttest_rel(c2para2performance[c][com_paris[1]], c2para2performance[c][com_paris[0]]).pvalue < 0.06:
                ax2.scatter(k, y[-1], marker = "P", s=200, c='red', zorder = 100)
            else:
                ax2.scatter(k, y[-1], marker = "o", s = 75, c='blue',alpha = 0.75)
            print(com_paris, stats.ttest_rel(c2para2performance[c][com_paris[1]], c2para2performance[c][com_paris[0]]).pvalue)
        ax2.plot([i for i in range(len(y))], y, linewidth = 4, color = 'blue', alpha = 0.75, label = com_paris[2], linestyle = com_paris[3])
    
    labels = [ '\n'.join(wrap(l, 20)) for l in c2str.values()]
    ax1.legend(labels, fontsize=fontsize-5, )
    ax1.set_xlabel("#Click",fontsize=fontsize)
    ax2.legend(fontsize=fontsize-5,)
    ax2.set_yticks([0.2,0.25,0.3,0.35],["$-5$","$0$","$5$","$10$"],fontsize=fontsize,alpha = 0.75,color='blue',rotation=30)

    plt.savefig('../results/parameter_udr.png',bbox_inches='tight')

def paint_performance_regarding_context_rdr():
    c2para2performance = rdr_bad_click()
    
    total_width, n = 0.8, 3
    width = total_width / n
    fontsize=26
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 10), sharey=True)
    plt.grid(alpha=0.5,axis='y')
    ax2 = ax1.twinx()
    
    ax1.set_xlabel('#Bad Click', fontsize=fontsize)
    ax1.set_ylabel('NDCG@10', fontsize=fontsize)
    ax1.set_ylim(0.1,0.9)
    colors = ['#9BA3BA','#e38c7a', '#C89598']
    ax1.set_xticks([0,1,2],['$0~(64.8\%)$','$1~(27.4\%)$','$\geq 2~(7.8\%)$'],fontsize=fontsize)
    ax1.set_yticks([0.2,0.4,0.6,0.8],[0.2,0.4,0.6,0.8],fontsize=fontsize,rotation=30)

    c2str = {
        'c': r"$\theta^{re,bs:c:p}$=1:5:0",
        'bs': r"$\theta^{re,bs:c:p}$=5:1:0",
        'best': r"$\theta^{re,p:c:bs}$=0:2:5",
    }

    for k,para in enumerate(['bs','best','c',]): # 
        x = []
        y = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
            print(len(c2para2performance[c][para]), y)
        plt.bar([i - total_width / 3 + width * k for i in range(len(y))], y, width=width-0.02, linewidth = 2, edgecolor = 'black', label = c2str[para], color = colors[k], alpha=0.5)

    for com_paris in [['best','bs',"5:1:0 vs 5:2:0",'dashed'],['best','c',"1:5:0 vs 5:2:0",'solid']]:
        y = []
        x = []
        for k,c in enumerate(c2para2performance.keys()):
            y.append(((np.mean(c2para2performance[c][com_paris[1]])-np.mean(c2para2performance[c][com_paris[0]]))/np.mean(c2para2performance[c][com_paris[0]])+0.05)*2)
            x.append(str(c))
            if stats.ttest_rel(c2para2performance[c][com_paris[1]], c2para2performance[c][com_paris[0]]).pvalue < 0.06:
                ax2.scatter(k, y[-1], marker = "P", s=200, c='red', zorder = 100)
            else:
                ax2.scatter(k, y[-1], marker = "o", s = 75, c='blue', alpha=0.75)
            print(com_paris, stats.ttest_rel(c2para2performance[c][com_paris[1]], c2para2performance[c][com_paris[0]]).pvalue)
        ax2.plot([i for i in range(len(y))], y, linewidth = 4, color = 'blue', alpha=0.75, label = com_paris[2], linestyle = com_paris[3])

    ax2.set_ylabel("IMP. (%)",fontsize=fontsize,color='blue',)
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, zorder=200)
    d = .015   
    ax2.plot((1 - d, 1 + d), (0.61-d, 0.61+d), **kwargs)  # top-right diagonal
    ax2.plot((1 - d, 1 + d), (0.65-d, 0.65+d), **kwargs)  # top-right diagonal
    ax2.plot((1, 1), (0.61, 0.65), color='white', clip_on=False, zorder=100,transform=ax2.transAxes,)  # top-right diagonal
    ax2.set_yticks([0,0.1,0.2,0.3,0.5,0.7,0.9],["$-5$","$0$","$5$","$10$","20","60","70"],fontsize=fontsize,color='blue',rotation=30)
    ax1.legend(fontsize=fontsize-5, )
    ax2.legend(fontsize=fontsize-5, )
    plt.savefig('../results/parameter_rdr.png',bbox_inches='tight')
    plt.show()

def paint_performance_regarding_context_udr_fixed_parameters1():
    c2para2performance = udr_click()

    fontsize=26
    fig, ax2= plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharey=True)
    ax12 = ax2.twinx()
    plt.grid(alpha=0.5,axis='y')
    colors = ['#e38c7a', '#dccfcc']
    c2str = {'best': r"$QE^{F_{\theta^{it}}(R^{bs},R^{c},R^p)}$",
            'ablation_no_bs': r"$QE^{F_{\theta^{it}}(R^{c},R^p)}$"
            }
    para2y = {}
    total_width, n = 0.8, 3
    width = total_width / n
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]) )
            x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        ax2.bar(x=[i - width/2 + width * k for i in range(len(y))], height = [item for item in y], width=width-0.02, label = c2str[para], color = colors[k], alpha=0.5,linewidth = 2, edgecolor = 'black',)
        print("ax2 y", y)

    y = []
    x = []
    for k,c in enumerate(c2para2performance.keys()):
        y.append(((np.mean(c2para2performance[c]['best'])-np.mean(c2para2performance[c]['ablation_no_bs']))/np.mean(c2para2performance[c]['ablation_no_bs']) +0.05)*1 + 0.2)
        x.append(str(c))
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.05:
            ax12.scatter(k, y[-1], marker = "P", s=200, c='red',zorder = 100)
        else:
            ax12.scatter(k, y[-1], marker = "o", s = 75, c='blue', alpha=0.75, zorder = 100)
    ax12.plot([i for i in range(len(y))], y, linewidth = 4, label = c2str[para], color = 'blue', alpha=0.75)

    ax2.set_ylim(0.15,0.55)  
            
    ax2.legend(fontsize=fontsize-3, )
    ax2.set_xlabel('#Click', fontsize=fontsize)
    ax2.set_xticks([0,1,2,],['$0~(51.5\%)$','$1~(20.9\%)$','$\geq 2~(19.3\%)$'], fontsize=fontsize)
    ax2.set_yticks([0.2,0.3,0.4,0.5],[0.2,0.3,0.4,0.5],fontsize=fontsize,rotation=30)
    ax12.set_yticks([0.2,0.25,0.3,0.35],['',0,5,10], fontsize=fontsize,color='blue',alpha=0.75,rotation=30)
    print(ax12.yaxis.get_majorticklabels())
    ax12.text(2.4,0.200,-5,fontsize=fontsize,color='blue',alpha=0.75,rotation=30)
    ax2.set_ylabel('NDCG@10', fontsize = fontsize)
    ax12.set_ylabel("IMP. (%)",fontsize=fontsize, color = 'blue', alpha = 0.75)
    plt.savefig('../results/udr1.png',bbox_inches='tight')

def paint_performance_regarding_context_rdr_fixed_parameters1():
    c2para2performance = rdr_bad_click()
    fontsize=26
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharey=True)
    ax2 = ax1.twinx()
    plt.grid(alpha=0.5,axis='y')
    ax1.set_xlabel('#Bad Click', fontsize=fontsize)
    colors = ['#e38c7a', '#dccfcc']
    ax1.set_xticks([0,1,2,],['$0~(64.8\%)$', '$1~(27.4\%)$','$\geq 2~(7.8\%)$'],fontsize=fontsize)
    ax1.set_ylim(0.1,0.9)
    ax1.set_ylabel('NDCG@10', fontsize = fontsize)
    ax1.set_yticks([0.2,0.4,0.6,0.8],[0.2,0.4,0.6,0.8], fontsize=fontsize,rotation=30)
    para2y = {}
    c2str = {'best': r"$F_{{\theta}^{it}}(R^{bs},R^{c},R^p)}$",
            'ablation_no_bs': r"$F_{\theta^{it}}(R^{c},R^p)}$"
            }
    total_width, n = 0.8, 3
    width = total_width / n
    for k,para in enumerate(['best', 'ablation_no_bs']): # 
        y = []
        x = []
        for c in c2para2performance.keys():
            y.append(np.mean(c2para2performance[c][para]))
            x.append(str(c))
        para2y[para] = y
        print(len(c2para2performance[c][para]), y)
        ax1.bar([i- width/2+k*width for i in range(len(y))], y, label = c2str[para], width=width-0.02, color = colors[k], alpha=0.5,linewidth = 2, edgecolor = 'black',)
    y = []
    x = []
    for k,c in enumerate(c2para2performance.keys()):
        v = (np.mean(c2para2performance[c]['best'])-np.mean(c2para2performance[c]['ablation_no_bs']))/np.mean(c2para2performance[c]['ablation_no_bs'])
        if v < 0.2:
            y.append((v+0.05)*2)
        else:
            y.append(v)
        x.append(str(c))
        if stats.ttest_rel(c2para2performance[c]['ablation_no_bs'], c2para2performance[c]['best']).pvalue < 0.01:
            ax2.scatter(k, y[-1], marker = "P", s=200, c='red',zorder=100)
    
    ax2.plot([i for i in range(len(y))], y, linewidth = 4, label = c2str[para], color = 'blue', alpha=0.75)
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, zorder=200)
    d = .015   
    ax2.plot((1 - d, 1 + d), (0.61-d, 0.61+d), **kwargs)  # top-right diagonal
    ax2.plot((1 - d, 1 + d), (0.65-d, 0.65+d), **kwargs)  # top-right diagonal
    ax2.plot((1, 1), (0.61, 0.65), color='white', clip_on=False, zorder=100,transform=ax2.transAxes,)  # top-right diagonal
    ax1.legend(fontsize=fontsize-3, )
    ax2.set_ylabel("IMP. (%)",fontsize=fontsize, color = 'blue', alpha = 0.75)
    ax2.set_ylim(0,0.9)
    ax2.set_yticks([0,0.1,0.2,0.3,0.5,0.7,0.9],[-5,0,5,10,20,60,70],fontsize=fontsize,color='blue',alpha=0.75,rotation=30)
    plt.savefig('../results/rdr1.png',bbox_inches='tight')

if __name__ == '__main__':
    # print images for performance comparaison in different scenarios with different parameters
    paint_performance_regarding_context_udr()
    paint_performance_regarding_context_rdr()

    # print images for performance comparaison in different scenarios
    paint_performance_regarding_context_udr_fixed_parameters1()
    paint_performance_regarding_context_rdr_fixed_parameters1()

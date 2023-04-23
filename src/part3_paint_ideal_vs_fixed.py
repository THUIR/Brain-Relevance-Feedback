import matplotlib.pyplot as plt
import seaborn as sns

def RDR_run():
    name_list = [r'$F_{\theta^{it}}(R^{bs},R^p)$',r'$F_{\theta^{it}}(R^{c},R^p)$',r'$F_{\theta^{it}}(R^{bs},R^c,R^p)$']
    num_list = [0.6973,0.7161,0.7693]
    err1=[0.2790,0.2668,0.2200]
    num_list1 =  [0.6973,0.7299,0.8413]
    err2=[0.2783,0.2668,0.2064]
    x =list(range(len(num_list)))
    total_width, n = 0.6, 2
    width = total_width / n
    # 绘制网格,alpha设置透明度
    # CEDFEF 92C2DD 4995C6 1663A9
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
    plt.ylim(0.35,1.0)
    plt.xlabel("Models",fontsize = 25)
    plt.ylabel("NDCG@10",fontsize = 25)
    plt.xlabel("Method")
    plt.ylabel("NDCG@10")
    plt.grid(alpha=0.5,axis='y')
    colors = ['#9BA3BA','#e38c7a', '#C89598']
    plt.bar(x, num_list, width=width-0.02, label='Fixed',fc = colors[1], alpha=0.5, linewidth = 2, edgecolor = 'black', ) # 
    for i in range(len(x)):
        x[i] = x[i] + width / 2
    plt.bar(x, num_list1, width=0, tick_label = name_list,fc = colors[1], alpha=0.8)
    for i in range(len(x)):
        x[i] = x[i] + width / 2
    plt.text(x[2]-width * 0.75, num_list1[2], "**", color='red', fontsize=25)
    plt.bar(x, num_list1, width=width-0.02, label='Ideal',fc =colors[0], alpha=0.5, linewidth = 2, edgecolor = 'black',) #sns.color_palette("rocket_r")[1]
    # plt.fill_between(x, y+y_std, y-y_std, alpha=0.2) 

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list_random, width=width, label='Random',tick_label = name_list,fc = sns.color_palette("light:b")[1], alpha=0.8)
    plt.axhline(0.4439,c="red", ls="--", lw=2, dashes=(5, 10), label='$BERT~(R^p)$',alpha=0.6)
    plt.legend(fontsize=25, ncol = 1, loc='upper left')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('../results/rdr.png', bbox_inches='tight')
    plt.cla()

def UDR_run():
    fig, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
    name_list = [r'$QE^{F_{\theta^{it}}(R^{bs},R^p)}$',r'$QE^{F_{\theta^{it}}(R^{c},R^p)}$',r'$QE^{F_{\theta^{it}}(R^{bs},R^c,R^p)}$']
    num_list = [0.3374,0.3690,0.3747]
    err1=[0.2790,0.2668,0.2200]
    num_list1 =  [0.3671,0.3845,0.4463]
    err2=[0.2783,0.2668,0.2064]
    x =list(range(len(num_list)))
    total_width, n = 0.6, 2
    width = total_width / n
    # 绘制网格,alpha设置透明度
    #  92C2DD  1663A9
    plt.ylim(0.3,0.5)
    plt.yticks([0.3,0.35,0.4,0.45,0.5])
    plt.xlabel("Models",fontsize = 25)
    plt.ylabel("NDCG@10",fontsize = 25)
    plt.xlabel("Method")
    plt.grid(alpha=0.5,axis='y')
    colors = ['#9BA3BA','#e38c7a', '#C89598']
    plt.bar(x, num_list, width=width-0.02, label='Fixed',fc = colors[1],alpha=0.5, linewidth = 2, edgecolor = 'black',)
    for i in range(len(x)):
        x[i] = x[i] + width/2
    plt.bar(x, num_list1, width=0, tick_label = name_list,fc = colors[1], alpha=0.8,)
    for i in range(len(x)):
        x[i] = x[i] + width/2
    plt.bar(x, num_list1, width=width-0.02, label='Ideal',fc = colors[0], alpha=0.5,linewidth = 2, edgecolor = 'black',)
    # plt.fill_between(x, y+y_std, y-y_std, alpha=0.2) 

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list_random, width=width, label='Random',tick_label = name_list,fc = sns.color_palette("light:b")[1], alpha=0.8)
    plt.axhline(0.3352,c="red", ls="--", lw=2, dashes=(5, 10), label='$BERT~(R^p)$',alpha=0.6)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.text(x[0]-width * 0.75, num_list1[0], "**", color='red', fontsize=25)
    plt.text(x[1]-width * 0.75, num_list1[1], "**", color='red', fontsize=25)
    plt.text(x[2]-width * 0.75, num_list1[2], "**", color='red', fontsize=25)
    plt.savefig('../results/udr.png', bbox_inches='tight')
    plt.cla()

if __name__ == '__main__':
    RDR_run()
    UDR_run()

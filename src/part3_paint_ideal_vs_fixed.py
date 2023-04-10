import matplotlib.pyplot as plt
import seaborn as sns

def RDR_run():
    name_list = ['$R^{bs}+R^p$','$R^c+R^p$','$R^{bs}+R^c+R^p$']
    num_list = [0.6973,0.7161,0.7695]
    err1=[0.2790,0.2668,0.2200]
    num_list1 =  [0.7294,0.7408,0.8416]
    err2=[0.2783,0.2668,0.2064]
    x =list(range(len(num_list)))
    total_width, n = 0.6, 2
    width = total_width / n
    # 绘制网格,alpha设置透明度
    # CEDFEF 92C2DD 4995C6 1663A9
    plt.ylim(0.35,1.0)
    plt.xlabel("Models",fontsize = 21)
    plt.ylabel("NDCG@10",fontsize = 21)
    plt.xlabel("Method")
    plt.ylabel("NDCG@10")
    plt.grid(alpha=0.5,axis='y')
    plt.bar(x, num_list, width=width, label='Fixed',fc = sns.color_palette("rocket_r")[0], alpha=0.8 ) # 
    for i in range(len(x)):
        x[i] = x[i] + width / 2
    plt.bar(x, num_list1, width=0, tick_label = name_list,fc = sns.color_palette("light:r")[3], alpha=0.8)
    for i in range(len(x)):
        x[i] = x[i] + width / 2
    plt.bar(x, num_list1, width=width, label='Adaptable',fc ='#e38c7a', alpha=0.75,) #sns.color_palette("rocket_r")[1]
    # plt.fill_between(x, y+y_std, y-y_std, alpha=0.2) 

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list_random, width=width, label='Random',tick_label = name_list,fc = sns.color_palette("light:b")[1], alpha=0.8)
    plt.axhline(0.4439,c="red", ls="--", lw=2, dashes=(5, 10), label='$Bert~(R^p)$',alpha=0.6)
    plt.legend(fontsize=18, ncol = 1, loc='upper left')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('../results/rdr.png', bbox_inches='tight')

def UDR_run():
    name_list = ['$QE^{R^{bs}+R^p}$','$QE^{R^c+R^p}$','$QE^{R^{bs}+R^c+R^p}$']
    num_list = [0.3487,0.3753,0.3797]
    err1=[0.2790,0.2668,0.2200]
    num_list1 =  [0.3750,0.3909,0.4521]
    err2=[0.2783,0.2668,0.2064]
    x =list(range(len(num_list)))
    total_width, n = 0.6, 2
    width = total_width / n
    # 绘制网格,alpha设置透明度
    #  92C2DD  1663A9
    plt.ylim(0.3,0.5)
    plt.yticks([0.3,0.35,0.4,0.45,0.5])
    plt.xlabel("Models",fontsize = 21)
    plt.ylabel("NDCG@10",fontsize = 21)
    plt.xlabel("Method")
    plt.grid(alpha=0.5,axis='y')
    plt.bar(x, num_list, width=width, label='Fixed',fc = '#CEDFEF',alpha=0.8 )
    for i in range(len(x)):
        x[i] = x[i] + width/2
    plt.bar(x, num_list1, width=0, tick_label = name_list,fc = sns.color_palette("light:r")[3], alpha=0.8)
    for i in range(len(x)):
        x[i] = x[i] + width/2
    plt.bar(x, num_list1, width=width, label='Adaptable',fc = '#bbc2d4', alpha=0.75,)
    # plt.fill_between(x, y+y_std, y-y_std, alpha=0.2) 

    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list_random, width=width, label='Random',tick_label = name_list,fc = sns.color_palette("light:b")[1], alpha=0.8)
    plt.axhline(0.3352,c="red", ls="--", lw=2, dashes=(5, 10), label='$Bert~(R^p)$',alpha=0.6)
    plt.legend(fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('../results/udr.png', bbox_inches='tight')

if __name__ == '__main__':
    RDR_run()
    UDR_run()
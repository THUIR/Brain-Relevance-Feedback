import json
from statsmodels.stats.anova import anova_lm, AnovaRM
from statsmodels.stats.multicomp import MultiComparison
import pandas as pd
from scipy.stats import bartlett, ttest_rel
from pingouin import rm_anova

df = pd.DataFrame(columns=['Treat', 'Value', 'd'])

u2info = json.load(open('../release/u2info.json'))

raw_method2method = {'equal':'Equal', 'bert-bs+click':r'BQE^{bs,c}', 'bert-click':r'BQE^{c}',}
method_list = [r'BQE^{bs,c}', r'BQE^{c}', 'Equal']

# significant testing

u2treat2v = {}
for u in sorted(u2info.keys()):
    if u.startswith('2_'):
        method2num = {r'BQE^{bs,c}':0, r'BQE^{c}':0, 'Equal':0}
        for raw_q in u2info[u]['raw_q2info'].keys():
            try:
                method2num[raw_method2method[u2info[u]['raw_q2info'][raw_q]['select']]] += 1
            except:
                # jiayudebug snippet start----------
                inputs = ''
                while inputs != 'continue':
                    try:
                        print(eval(inputs))
                    except Exception as e:
                        print(e)
                    inputs = input()
                # jiayudebug snippet end-------------
        u2treat2v[u] = method2num
        for i in range(len(method_list)):
            df.loc[len(df)] = [method_list[i], method2num[method_list[i]], u]
        
treat2v_list = {}
for u in u2treat2v.keys():
    for treat in u2treat2v[u].keys():
        if treat not in treat2v_list.keys():
            treat2v_list[treat] = []
        treat2v_list[treat].append(u2treat2v[u][treat])

print(bartlett(treat2v_list[r'BQE^{bs,c}'], treat2v_list[r'BQE^{c}'], treat2v_list['Equal']))

print(AnovaRM(df,'Value', 'd', within=['Treat']).fit())
df['Value'] = [float(item) for item in df['Value']]
print(rm_anova(dv='Value', within='Treat', subject='d',data=df, correction=True))

mc = MultiComparison(df['Value'],df['Treat'])
tukey_result = mc.allpairtest(ttest_rel, alpha = 0.05)
print(tukey_result[0])

# result painting

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def list_normalized(arr1):
    return np.array(arr1) / np.sum(arr1)

data = []
for u in u2treat2v.keys():
    data.append(list_normalized(list(u2treat2v[u].values())))
data = np.array(data).T

plt.figure(figsize=(20, 10))

plot=sns.heatmap(data)

plt.yticks([0.5,1.5,2.5], method_list)
plt.xlabel('User ID')

plt.savefig('../results/part1_online_pairwise_hetmap.jpg', bbox_inches = 'tight')

plt.cla()
plt.figure(figsize=(20, 10))

data = []
for u in u2treat2v.keys():
    data.append(list(u2treat2v[u].values()))
data = np.array(data).T

from aquarel import load_theme
# https://huaban.com/pins/4476502099

x, y1, y2 = [i for i in range(len(data[0]))], [data[0][i] for i in range(len(data[0]))], [-data[1][i] for i in range(len(data[0]))]

def sorted_according_to(x1, x2):
    x_merge = sorted([[x1[i], x2[i]] for i in range(len(x1))], key = lambda v: v[1], reverse = True)
    return [item[0] for item in x_merge]
y2 = sorted_according_to(y2, y1)
x_sorted = sorted_according_to(x, y1)
y1 = sorted_according_to(y1, y1)


plt.bar(x, y1, color='#b2c3b3', label = '$BQE^{bs,c}$')
plt.bar(x, y2, color='#be9997', label = '$BQE^{c}$')

plt.yticks([-5,-10,0,5,10,15,20,25,30],[5,10,0,5,10,15,20,25,30], fontsize = 20)
plt.xticks(x, [x_sorted[i]+1 for i in range(12)], fontsize = 20)
plt.xlabel('Participant ID', fontsize = 20)
plt.ylabel('Number', fontsize = 20)
plt.legend(fontsize = 20)

# https://color.adobe.com/zh/explore

plt.savefig('../results/part1_online_pairwise_bar.jpg', bbox_inches = 'tight')

plt.cla()
plt.figure(figsize=(20, 10))

data = []
for u in u2treat2v.keys():
    data.append(list(u2treat2v[u].values()))
data = np.array(data).T

from aquarel import load_theme
# https://huaban.com/pins/4476502099

x, y1, y2 = [i for i in range(len(data[0]))], [data[0][i] for i in range(len(data[0]))], [data[1][i] for i in range(len(data[0]))]

y1, y2 = [y1[i]/(y1[i]+y2[i]) for i in range(len(y1))], [-y2[i]/(y1[i]+y2[i]) for i in range(len(y1))]

def sorted_according_to(x1, x2):
    x_merge = sorted([[x1[i], x2[i]] for i in range(len(x1))], key = lambda v: v[1], reverse = True)
    return [item[0] for item in x_merge]
y2 = sorted_according_to(y2, y1)
x_sorted = sorted_according_to(x, y1)
y1 = sorted_according_to(y1, y1)
x_sorted = [item+20 for item in x_sorted]

plt.bar(x, y1, color='#b2c3b3', label = r"$F_{\theta}^{re}(R^{bs},R^c,R^p)$")
plt.bar(x, y2, color='#be9997', label = r"$F_{\theta}^{re}(R^c,R^p)$")
fontsize = 25

plt.yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0],[1.0,0.75,0.5,0.25,0,0.25,0.5,0.75,1.0], fontsize = fontsize)
plt.xticks(x, [x_sorted[i]+1 for i in range(12)], fontsize = fontsize)
plt.xlabel('Participant ID', fontsize = fontsize)
plt.ylabel('Ratio', fontsize = fontsize)
plt.legend(fontsize = fontsize)

# https://color.adobe.com/zh/explore

plt.savefig('../results/part1_online_pairwise_bar_normalized.jpg', bbox_inches = 'tight')


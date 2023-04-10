import json
import pandas as pd
import numpy as np
import copy

def true_data(mode, task = 'udr'):
    if task== 'udr':
        true_data_result_list = json.load(open(f'../results/simulating/true_data_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/true_data.{mode}.json'))
    else:
        true_data_result_list = json.load(open(f'../results/simulating/rdr_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/rdr.{mode}.json'))
    i2subset = {}
    info_list = []
    for i in range(len(true_data_list)):
        q = true_data_list[i]['q']
        i2subset[i] = true_data_list[i]['subset']
        # I don't know the intent, choose average best performance
        for intent in true_data_result_list[i].keys():
            for subset in true_data_result_list[i][intent].keys():
                for para in true_data_result_list[i][intent][subset].keys():    
                    performance = np.mean(true_data_result_list[i][intent][subset][para])
                    info_list.append([i, q, intent, subset, para, performance])
    true_data_df = pd.DataFrame(info_list, columns = ['i', 'q','intent', 'subset', 'para', 'performance'])
    true_data_df_series = true_data_df.drop(columns = ['intent']).groupby(["i", "q","subset",'para']).sum().reset_index()
    max_flag = true_data_df_series.sort_values(by=["performance"],ascending=False).groupby(["i", "q","subset"]).head(1).drop(columns = ['para'])
    max_flag['flag'] = True
    true_data_merge = true_data_df_series.merge(max_flag,how='left')
    true_data_new = true_data_merge.loc[true_data_merge.flag==True]
    params = true_data_new.groupby(["i", 'q', 'subset',]).para.apply(lambda x:list(x))
    params = params.reset_index()

    def get_str_mean(str_list):
        float_list = [[float(v) for v in item.split('_')] for item in str_list]
        return json.dumps(np.mean(float_list, axis=0).tolist())
    params.para = pd.DataFrame(params.para).applymap(get_str_mean)

    def subset_id2ddd(i, subset_id, i2subset):
        subset_list = i2subset[i]
        subset = subset_list[int(subset_id)]
        return sorted(subset)
    
    def ddd_merge(ddd, dic, para):
        for i in range(len(ddd)):
            if ddd[i] not in dic.keys():
                dic[ddd[i]] = {}
            dic = dic[ddd[i]]
        dic['para'] = para

    q2subset2para = {}
    for i in range(len(params)):
        q = params.loc[i]['q']
        ddd = subset_id2ddd(params.loc[i]['i'], params.loc[i]['subset'], i2subset)
        para = json.loads(params.loc[i]['para'])
        if q not in q2subset2para.keys():
            q2subset2para[q] = {}
        ddd_merge(ddd, q2subset2para[q], para)
    
    json.dump(q2subset2para, open(f'../results/simulating/q2subset2para.{mode}.json', 'w'))

def test_click(true_data_list, true_data_result_list):
    i2subset = {}
    click2num = {}
    for i in range(len(true_data_list)):
        q = true_data_list[i]['q']
        i2subset[i] = true_data_list[i]['subset']
        # I don't know the intent, choose average best performance
        for intent in true_data_result_list[i].keys():
            for subset in true_data_result_list[i][intent].keys():
                for para in true_data_result_list[i][intent][subset].keys():
                    click = int(para.split('_')[-1])
                    if click not in click2num.keys():
                        click2num[click] = 0
                    click2num[click] += 1
    print('click2num', click2num)
    return click2num


def true_data_click(mode, task = 'udr'):
    if task== 'udr':
        true_data_result_list = json.load(open(f'../results/simulating/true_data_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/true_data.{mode}.json'))
    else:
        true_data_result_list = json.load(open(f'../results/simulating/rdr_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/rdr.{mode}.json'))
    # test_click(true_data_list, true_data_result_list)
    i2subset = {}
    info_list = []
    for i in range(len(true_data_list)):
        q = true_data_list[i]['q']
        i2subset[i] = true_data_list[i]['subset']
        # I don't know the intent, choose average best performance
        for intent in true_data_result_list[i].keys():
            for subset in true_data_result_list[i][intent].keys():
                for para in true_data_result_list[i][intent][subset].keys():    
                    performance = np.mean(true_data_result_list[i][intent][subset][para])
                    click = int(float(para.split('_')[-1]))
                    para ='_'.join(para.split('_')[:-1])
                    info_list.append([i, q, intent, subset, para, performance, click])
    true_data_df = pd.DataFrame(info_list, columns = ['i', 'q','intent', 'subset', 'para', 'performance','click'])
    true_data_df_series = true_data_df.drop(columns = ['intent']).groupby(["i", "q","subset",'para','click']).sum().reset_index()
    max_flag = true_data_df_series.sort_values(by=["performance"],ascending=False).groupby(["i", "q","subset",'click']).head(1).drop(columns = ['para'])
    max_flag['flag'] = True
    true_data_merge = true_data_df_series.merge(max_flag,how='left')
    true_data_new = true_data_merge.loc[true_data_merge.flag==True]
    params = true_data_new.groupby(["i", 'q', 'subset','click']).para.apply(lambda x:list(x))
    params = params.reset_index()

    def get_str_mean(str_list):
        float_list = [[float(v) for v in item.split('_')] for item in str_list]
        return json.dumps(np.mean(float_list, axis=0).tolist())
    params.para = pd.DataFrame(params.para).applymap(get_str_mean)

    def subset_id2ddd(i, subset_id, i2subset):
        subset_list = i2subset[i]
        subset = subset_list[int(subset_id)]
        return sorted(subset)
    
    def ddd_merge(ddd, dic, para, click):
        for i in range(len(ddd)):
            if ddd[i] not in dic.keys():
                dic[ddd[i]] = {}
            dic = dic[ddd[i]]
        if 'para' not in dic.keys():
            dic['para'] = {}
        dic['para'][str(click)] = para

    q2subset2click2para = {}
    for i in range(len(params)):
        q = params.loc[i]['q']
        ddd = subset_id2ddd(params.loc[i]['i'], params.loc[i]['subset'], i2subset)
        click = params.loc[i]['click']
        para = json.loads(params.loc[i]['para'])
        if q not in q2subset2click2para.keys():
            q2subset2click2para[q] = {}
        ddd_merge(ddd, q2subset2click2para[q], para, click)
    if args.task == 'rdr':
        mode = 'rdr.' + mode
    json.dump(q2subset2click2para, open(f'../results/simulating/q2subset2click2para.{mode}.json', 'w'))

def bug_true_data(task, mode):
    if task == 'udr':
        true_data_result_list = json.load(open(f'../results/simulating/true_data_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/true_data.{mode}.json'))
    else:
        true_data_result_list = json.load(open(f'../results/simulating/rdr_result_list.{mode}.json'))
        true_data_list = json.load(open(f'../results/simulating/rdr.{mode}.json'))
    i2subset = {}
    info_list = []
    for i in range(len(true_data_list)):
        q = true_data_list[i]['q']
        i2subset[i] = true_data_list[i]['subset']
        # I don't know the intent, choose average best performance
        for intent in true_data_result_list[i].keys():
            for subset in true_data_result_list[i][intent].keys():
                for para in true_data_result_list[i][intent][subset].keys():    
                    performance = np.mean(true_data_result_list[i][intent][subset][para])
                    info_list.append([i, q, intent, subset, para, performance])
    true_data_df = pd.DataFrame(info_list, columns = ['i', 'q','intent', 'subset', 'para', 'performance'])
    true_data_df_series = true_data_df
    max_flag = true_data_df_series.sort_values(by=["performance"],ascending=False).groupby(["i", "q",'intent',"subset"]).head(1).drop(columns = ['para'])
    max_flag['flag'] = True
    true_data_merge = true_data_df_series.merge(max_flag,how='left')
    true_data_new = true_data_merge.loc[true_data_merge.flag==True]
    params = true_data_new.groupby(["i", 'q', 'intent','subset',]).para.apply(lambda x:list(x))
    params = params.reset_index()

    def get_str_mean(str_list):
        float_list = [[float(v) for v in item.split('_')] for item in str_list]
        return json.dumps(np.mean(float_list, axis=0).tolist())
    params.para = pd.DataFrame(params.para).applymap(get_str_mean)

    def subset_id2ddd(i, subset_id, i2subset):
        subset_list = i2subset[i]
        subset = subset_list[int(subset_id)]
        return sorted(subset)
    
    def ddd_merge(ddd, dic, para):
        for i in range(len(ddd)):
            if ddd[i] not in dic.keys():
                dic[ddd[i]] = {}
            dic = dic[ddd[i]]
        dic['para'] = para

    q2intent2subset2para = {}
    for i in range(len(params)):
        q = params.loc[i]['q']
        intent = params.loc[i]['intent']
        ddd = subset_id2ddd(params.loc[i]['i'], params.loc[i]['subset'], i2subset)
        para = json.loads(params.loc[i]['para'])
        if q not in q2intent2subset2para.keys():
            q2intent2subset2para[q] = {}
        if intent not in q2intent2subset2para[q].keys():
            q2intent2subset2para[q][intent] = {}
        ddd_merge(ddd,  q2intent2subset2para[q][intent], para)
    
    json.dump(q2intent2subset2para, open(f'../results/simulating/q2intent2subset2para.{task}.{mode}.json', 'w'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',type=str, help= 'which mode to use, ', required=False, default = 'exp_pre.roberta_soft')
    parser.add_argument('-click',type=str, help= 'whether add click signals into the construction of search context', required=False, default = 'no')
    parser.add_argument('-task',type=str, help= 'choose from udr and rdr', required=False, default = 'udr')
    args = parser.parse_args()
    if args.click == 'no':
        true_data(mode = args.mode, task = args.task)
    else:
        true_data_click(mode = args.mode, task = args.task)


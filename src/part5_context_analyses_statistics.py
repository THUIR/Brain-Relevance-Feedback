import json
from scipy import stats
import numpy as np
from part5_context_analyses_utils import udr_click, udr_length, rdr_click, rdr_bad_click, rdr_length

def performance_table(u2result_list):
    def print_result2(method, result_dic):
        print(method)
        for key in result_dic.keys():
            print(f'{format(np.mean(result_dic[key]),".4f")}/{format(np.std(result_dic[key]),".4f")}', end = '\t')
        print()

    def show_performance(u2result_list):
        result_list = {}
        for u in u2result_list.keys():
            for method in u2result_list[u].keys():
                if method not in result_list.keys():
                    result_list[method] = {}
                for metric in u2result_list[u][method].keys():
                    if metric not in result_list[method].keys():
                        result_list[method][metric] = []
                    result_list[method][metric] += (u2result_list[u][method][metric])
        for method in result_list.keys():
            if len(result_list[method]['ndcg@1']) == 0 or np.isnan(result_list[method]['ndcg@1'][0]):
                continue
            print_result2(method, result_list[method])
        for test_pairs in [['bqe(bs+c-s)', 'bqe(c-s)']]:
            if test_pairs[0] in result_list.keys() and test_pairs[1] in result_list.keys():
                print(f'{test_pairs[0]} vs {test_pairs[1]}: ', stats.ttest_rel(result_list['bqe(bs+c-s)']['ndcg@10'], result_list['bqe(c-s)']['ndcg@10']).pvalue)

    show_performance(u2result_list)

def rdr_rule_based_performance():
    print('rdr rule-based adaptable parameters performance vs fixed parameters performance w.r.t. num of bad clicks:')
    c2para2performance = rdr_bad_click()
    rule_based = c2para2performance[0]['best'] + c2para2performance[1]['bs'] + c2para2performance[2]['bs']
    fixed = c2para2performance[0]['best'] + c2para2performance[1]['best'] + c2para2performance[2]['best']
    print('rule-based: ', np.mean(rule_based), 'fixed: ', np.mean(fixed), 't-test: ', stats.ttest_rel(rule_based, fixed))

    print('rdr rule-based adaptable parameters performance vs fixed parameters performance w.r.t. session length:')
    c2para2performance = rdr_length()
    rule_based = c2para2performance[1]['best'] + c2para2performance[2]['best'] + c2para2performance[3]['best'] + c2para2performance[4]['bs'] 
    fixed = c2para2performance[1]['best'] + c2para2performance[2]['best'] + c2para2performance[3]['best'] + c2para2performance[4]['best'] 
    print('rule-based: ', np.mean(rule_based), 'fixed: ', np.mean(fixed), 't-test: ', stats.ttest_rel(rule_based, fixed))

def udr_rule_based_performance():
    print('udr rule-based adaptable parameters performance vs fixed parameters performance w.r.t. num of bad clicks:')
    c2para2performance = udr_click()
    rule_based = c2para2performance[0]['bs'] + c2para2performance[1]['best'] + c2para2performance[2]['best']
    fixed = c2para2performance[0]['best'] + c2para2performance[1]['best'] + c2para2performance[2]['best']
    print('rule-based: ', np.mean(rule_based), 'fixed: ', np.mean(fixed), 't-test: ', stats.ttest_rel(rule_based, fixed))

def udr_performance_difference():
    c2para2performance = udr_length()
    c2para2performance2 = {0:{}, 1:{}}
    for c in c2para2performance.keys():
        if int(c) <= 4 and int(c) >= 1:
            r = 0
        else:
            r = 1
        for para in c2para2performance[c].keys():
            if para not in c2para2performance2[r].keys():
                c2para2performance2[r][para] = []
            c2para2performance2[r][para] += c2para2performance[c][para]
    print('---------split h < 4 vs h > 4-------------')
    for c in c2para2performance2.keys():
        print(c, (np.mean(c2para2performance2[c]['best']) - np.mean(c2para2performance2[c]['ablation_no_bs']))/np.mean(c2para2performance2[c]['ablation_no_bs']), stats.ttest_rel(c2para2performance2[c]['best'], c2para2performance2[c]['ablation_no_bs']))

def num_bad_click():
    num_click = 0
    num_bad_click = 0
    c2para2performance = rdr_bad_click()
    for c in c2para2performance.keys():
        if c == 0:
            num_click += len(c2para2performance[c]['bs'])
        else:
            num_click += len(c2para2performance[c]['bs'])
            num_bad_click += len(c2para2performance[c]['bs']) * c
    print("num_click", num_click)
    print("num_bad_click", num_bad_click)

if __name__ == '__main__':
    # print performance table
    rdr_results = json.load(open('../results/part3_asynchronize/part3_asy.json'))
    udr_results = json.load(open('../results/part3_synchronoize/part3_syn.json'))
    print('rdr performance')
    performance_table(rdr_results)
    print()
    print('udr performance')
    performance_table(udr_results)
    print()
    # analyze num of bad clicks
    num_bad_click()
    print()
    # rule-based performance computing
    rdr_rule_based_performance()
    print()
    udr_rule_based_performance()
    print()
    # analyze udr performance difference w.r.t. session length and num of clicks
    udr_performance_difference()
    print()

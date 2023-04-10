from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np

def softmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs

def combine_array(arr1, arr2, alpha):
    return [alpha*arr1[i]+(1-alpha)*arr2[i] for i in range(len(arr2))]

# print formated results 
def print_result(method, result_dic):
    print(method)
    for key in result_dic.keys():
        print(f'{key}: {format(np.mean(result_dic[key]),".4f")} + {format(np.std(result_dic[key]),".4f")}', end = ' ')
    print()

# print results with another format
def print_result2(method, result_dic):
    print(method)
    for key in result_dic.keys():
        print(f'{format(np.mean(result_dic[key]),".4f")}/{format(np.std(result_dic[key]),".4f")}', end = '\t')
    print()

# add result for a data sample into the result dict
def add_result(y_true, y_pred, result_dic, prev_len = 0, information = None):
    def thread_thord(y_true, thord = 0):
        re = []
        for i in range(len(y_true)):
            if y_true[i] > thord:
                re.append(1)
            else:
                re.append(0)
        return re
    try:
        for k in [1,3,5,10]:
            result_dic[f'ndcg@{k}'].append(ndcg_score([y_true],[y_pred],k=k))
        result_dic['map'].append(average_precision_score(thread_thord(y_true), y_pred))
    except Exception as e:
        print('caculating evaluation metcis leads to error! ', e, y_pred, y_true)
        # jiayudebug snippet start----------
        inputs = ''
        while inputs != 'continue':
            try:
                print(eval(inputs))
            except Exception as e:
                print(e)
            inputs = input()
        # jiayudebug snippet end-------------
        pass
    if information != None:
        if 'information' not in result_dic.keys():
            result_dic['information'] = []
        result_dic['information'].append(information)

# get sum of metrics
def get_metric_sum(y_true, y_pred, prev_len = 0):
    def thread_thord(y_true, thord = 0):
        re = []
        for i in range(len(y_true)):
            if y_true[i] > thord:
                re.append(1)
            else:
                re.append(0)
        return re
    re = 0
    for k in [1,3,5]:
        re += ndcg_score([y_true],[y_pred],k=k)
    re += average_precision_score(thread_thord(y_true), y_pred)
    return re

def bert_qm_init(q2d2score, q, now_d_list, future_d_list, args, d2score = None):
    now_d_list_bert_score = [[d,q2d2score[q][d]['score']] for d in now_d_list]
    pse_gamma = 1 if 'pse_gamma' not in vars(args).keys() else args.pse_gamma
    if d2score == None:
        now_d_list_bert_score_select = sorted([[now_d_list_bert_score[i][0], now_d_list_bert_score[i][1]] for i in range(len(now_d_list_bert_score))], key = lambda v: v[1], reverse = True)
    else:
        
        now_d_list_bert_score_select = sorted([[item[0], pse_gamma * item[1] + args.eeg_gamma * d2score[item[0]][0] + args.click_gamma * d2score[item[0]][1]] for i, item in enumerate(now_d_list_bert_score)], key=lambda v: v[1], reverse=True)

    future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
    now_d_list_bert_score_select = now_d_list_bert_score_select[:args.kd]
    now_d_list_set = set([item[0] for item in now_d_list_bert_score_select])
    now_d_list_bert_split_score = [[d,pos,score] for d in now_d_list for pos,score in q2d2score[q][d]['split_score'].items() if d in now_d_list_set]
    if d2score == None:
        pass
    else:
        now_d_list_bert_split_score = [[now_d_list_bert_split_score[i][0], now_d_list_bert_split_score[i][1], pse_gamma * now_d_list_bert_split_score[i][2] + args.click_gamma * d2score[now_d_list_bert_split_score[i][0]][1] + args.eeg_gamma * d2score[now_d_list_bert_split_score[i][0]][0]] for i in range(len(now_d_list_bert_split_score))]
    now_d_list_bert_split_score = sorted(now_d_list_bert_split_score, key = lambda v: v[2], reverse = True)[:args.kc]
   
    return future_score1, now_d_list_bert_split_score

def position_softmax(arr, pos):
    arr_pos = [item[pos] for item in arr]
    arr_pos = softmax(arr_pos)
    for i in range(len(arr)):
        arr[i][pos] = arr_pos[i]
    return arr

def bert_qm_procedure(q, now_d_list_bert_split_score, future_d_list, q2d2d2score,):
    future_score2 = []
    now_d_list_bert_split_score = position_softmax(now_d_list_bert_split_score, 2)
    for d in future_d_list:
        tmp_score = np.sum([now_d_list_bert_split_score[i][2] * q2d2d2score[q][now_d_list_bert_split_score[i][0]][d]['split_score'][now_d_list_bert_split_score[i][1]] for i in range(len(now_d_list_bert_split_score))])
        future_score2.append(tmp_score)
    return future_score2

def bert_qm_all(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score = None):
    future_score1, now_d_list_bert_split_score = bert_qm_init(q2d2score, q, now_d_list, future_d_list, args, d2score)
    future_score2 = bert_qm_procedure(q, now_d_list_bert_split_score, future_d_list, q2d2d2score,)
    return future_score1, future_score2

def bert_qm_init_baseline(q2d2score, q, now_d_list, future_d_list, args, d2score = None):
    now_d_list_bert_score = [[d,q2d2score[q][d]['score']] for d in now_d_list]
    if d2score == None:
        now_d_list_bert_score_select = sorted([[now_d_list_bert_score[i][0], now_d_list_bert_score[i][1]] for i in range(len(now_d_list_bert_score))], key = lambda v: v[1], reverse = True)
    else:
        now_d_list_bert_score_select = sorted([[item[0], item[1] + args.gamma * d2score[item[0]]] for i, item in enumerate(now_d_list_bert_score)], key=lambda v: v[1], reverse=True)

    future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
    now_d_list_bert_score_select = now_d_list_bert_score_select[:args.kd]
    now_d_list_set = set([item[0] for item in now_d_list_bert_score_select])
    now_d_list_bert_split_score = [[d,pos,score] for d in now_d_list for pos,score in q2d2score[q][d]['split_score'].items() if d in now_d_list_set]
    if d2score == None:
        pass
    else:
        now_d_list_bert_split_score = [[now_d_list_bert_split_score[i][0], now_d_list_bert_split_score[i][1], now_d_list_bert_split_score[i][2] + args.gamma * d2score[now_d_list_bert_split_score[i][0]]] for i in range(len(now_d_list_bert_split_score))]
    now_d_list_bert_split_score = sorted(now_d_list_bert_split_score, key = lambda v: v[2], reverse = True)[:args.kc]
   
    return future_score1, now_d_list_bert_split_score

def bert_qm_all_baseline(q2d2score, q, now_d_list, future_d_list, args, q2d2d2score, d2score = None):
    future_score1, now_d_list_bert_split_score = bert_qm_init_baseline(q2d2score, q, now_d_list, future_d_list, args, d2score)
    future_score2 = bert_qm_procedure(q, now_d_list_bert_split_score, future_d_list, q2d2d2score,)
    return future_score1, future_score2

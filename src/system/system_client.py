import json
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../system/')
from system.utils import position_softmax, bert_qm_procedure, combine_array, bert_qm_init
from system.system_display import Displayer
from sklearn.metrics import roc_auc_score
import threading
from system.feature_extraction import preprocessed
from classifier import Base_models
import random
import os
import joblib
from sklearn import preprocessing

# 不用normalizer了，注意之后可以看看normalizer会不会更好
class Eeg_classifier():
    def topography_invariant(self, data):
        x = []
        for item in data:
            x.append([feature for single_channel_data in item for feature in single_channel_data])
        return x

    def label2binary(self, label_list):
        label_list = [1 if item > 1 else 0 for item in label_list]
        return label_list

    def get_metric(self, y_pred, y_true, cut_lr, epoch_size = 25):
        re = []
        y_pred = np.array(y_pred[cut_lr[0]: cut_lr[1]])
        y_true = np.array(y_true[cut_lr[0]: cut_lr[1]])

        leave_count = 0
        for i in range(0, len(y_true), epoch_size):
            l  = i - leave_count * epoch_size
            if np.std(y_true[l:i+epoch_size]) > 0:
                re.append(roc_auc_score(y_true[l:i+epoch_size], y_pred[l:i+epoch_size]))
                leave_count = 0
            else:
                leave_count += 1       
        return np.mean(re)

    def __init__(self, u = ''):
        # load from ~/online/online/src/meta/models/online_svm.model use the normalized mode
        if os.path.exists(f'../release/para/online_svm_{u}.model'):
            self.general_model = joblib.load(f'../release/para/online_svm_{u}.model')
        else:
            self.general_model = joblib.load('../release/para/online_svm.model')
        self.decrease_rate = 0.9
        self.x = []
        self.y = []
        self.y_true = []
        self.specific_model = Base_models('svm', {"lr":0.1, "epoches":200, "depth":3, "split":10, 'features':4, "leaf":3,"subsample":0.95, "verbose":False, 'svm_c':0.1, 'gamma':10})
        self.trainer = None
        self.shot = 0
        self.start_rate = 1.0
        self.y_pred = []
        self.update_epoch = 40
        self.current_epoch = 0
        self.y_pred_general = []
        self.y_pred_specific = []
        self.normalized = True
        self.q2d2info = {}
        self.y_true_std = False
    
    def train(self,):
        # filter None!
        if self.y_true_std or np.std(self.y_true) > 0:
            self.y_true_std = True
            self.specific_model.fit(self.x, self.y_true)
        if len(self.x) >= 120 and self.start_rate > 0:
            if self.get_metric(self.y_pred_general, self.y_true, [len(self.x)-100,len(self.x)]) < self.get_metric(self.y_pred_specific, self.y_true, [len(self.x)-100,len(self.x)]) and self.start_rate > 0.5:
                self.start_rate = 0.75
            elif self.get_metric(self.y_pred_general, self.y_true, [len(self.x)-100,len(self.x)]) + 0.02 < self.get_metric(self.y_pred_specific, self.y_true, [len(self.x)-100,len(self.x)]) and self.start_rate > 0:
                self.start_rate = 0

    def add_label(self, question_id, y):
        self.shot += len(y)
        filtered_y = []
        if question_id not in self.q2d2info.keys():
            return
        for d in self.q2d2info[question_id].keys():
            if len(y) > d:
                filtered_y.append(y[d])
                self.x += self.q2d2info[question_id][d]['x']
                self.y_pred += self.q2d2info[question_id][d]['y_pred']
                self.y_pred_general += self.q2d2info[question_id][d]['y_pred_general']
                self.y_pred_specific += self.q2d2info[question_id][d]['y_pred_specific']
            else:
                print('length of y not match')

        y = [filtered_y[i//2] for i in range(len(filtered_y)*2)]
        self.y += y # shot数量和data数量不一致！
        self.y_true += self.label2binary(y)
        self.current_epoch += len(y) * 2
        if self.current_epoch >= self.update_epoch:
            if self.trainer != None:
                self.trainer.join()
            self.trainer = threading.Thread(target = self.train)
            self.trainer.start()
    
    def predict(self,x):
        self.current_epoch += len(x)
        tmp_rate = self.start_rate * self.decrease_rate ** (len(self.y) / 50)
        y_pred_general = self.general_model.predict_proba(x)
        if self.y_true_std:
            y_pred_specific = self.specific_model.predict_proba(x)
        else:
            y_pred_specific = y_pred_general
        y_pred = []
        for i in range(len(x)):
            if self.shot > 40:
                y_pred.append(y_pred_general[i] * tmp_rate + y_pred_specific[i] * (1 - tmp_rate))
            else:
                y_pred.append(y_pred_general[i])

        return y_pred, y_pred_general, y_pred_specific

    def add_data(self, question_id, doc_id,  x, anno = 0, land = False):
        # wait until the mopdel prepared
        if self.trainer != None:
            self.trainer.join()
        if type(x) == dict:
            x = x['fs']
            my_std = preprocessing.StandardScaler()
            for i in range(len(x)):
                x[i] = my_std.fit_transform(x[i]).tolist()
        else:
            x = preprocessed(x, self.normalized, None)
        if type(x) == type(None):
            return None
        x = self.topography_invariant(x)
        y_pred, y_pred_general, y_pred_specific = self.predict(x)
        if question_id not in self.q2d2info.keys():
            self.q2d2info[question_id] = {}
        if land:
            self.x += x
            self.y_true += [anno for i in range(len(x))]
            self.y_pred += y_pred
            self.y_pred_general += y_pred_general
            self.y_pred_specific += y_pred_specific
        else:
            if doc_id not in self.q2d2info[question_id].keys():
                self.q2d2info[question_id][doc_id] = {
                    'y_pred': y_pred,
                    'y_pred_general': y_pred_general,
                    'y_pred_specific': y_pred_specific,
                    'x': x,
                }
        return np.mean(y_pred), np.mean(y_pred_general), np.mean(y_pred_specific)
    
    def get_performance(self,):
        return roc_auc_score(self.y_true,self.y_pred[:len(self.y_true)]), roc_auc_score(self.y_true,self.y_pred_general[:len(self.y_true)]), roc_auc_score(self.y_true,self.y_pred_specific[:len(self.y_true)])

# 模拟eeg classifier，用于我们的虚假用户实验（不带脑电帽子的）
class Simulator_classifier():
    def __init__(self,):
        self.eeg_mean = 0.4482
        self.eeg_std = 0.1123
        self.para = {'anno':{0:0,1:0},'gd':{0:0,1:0}}
        self.para['anno'][0] = {'loc': 0.4294, 'scale': 0.1263}
        self.para['anno'][1] = {'loc': 0.47, 'scale': 0.1259}
        self.para['gd'][1] = {'loc': 0.4784, 'scale': 0.1317}
        self.para['gd'][0] = {'loc': 0.4210, 'scale': 0.1208}
        self.shot = 0
        self.mod = 100
        self.general_std = 1.3
        self.decrease_rate = 0.92    
    def add_data(self, question_id, doc_id, x, anno, land=False):
        self.shot += 1
        scale = self.para['anno'][anno]['scale'] * self.general_std * self.decrease_rate ** (self.shot / self.mod)
        return np.random.normal(self.para['anno'][anno]['loc'], scale)
    def add_label(self, question_id, y):
        pass

class Simple_ranker():
    def __init__(self, mode = 'eeg'):
        self.mode = mode
        self.m = 20
        self.anno = json.load(open('../system/data/total_info_no_filtered.json'))
        self.q2d2score = json.load(open(f'../system/data/q2d2score_{self.m}.json'))
        # share the same eeg classifier
        self.eeg_classifier = None
    def init_docs(self,q,d_list,intent):
        self.q = q
        self.d_list = d_list
        self.d2score = {}
        self.now_d_list = []
        self.future_d_list = d_list
        self.intent = intent
        # rerank according to bert score
        self.future_d_list = [[d, self.q2d2score[q][d]['score']] for d in self.future_d_list]
        self.future_d_list = sorted(self.future_d_list, key= lambda v:v[1], reverse = True)
        self.future_d_list = [d[0] for d in self.future_d_list]
    def update_status(self, question_id, pos, d, click, eeg, anno =0):
        if d not in self.now_d_list:
            self.now_d_list.append(d)
            del self.future_d_list[self.future_d_list.index(d)]
        self.get_satisfaction(question_id, pos, eeg, anno)
    def update_labels(self,question_id, y):
        if self.eeg_classifier != None:
            self.eeg_classifier.add_label(question_id, y)
    def get_next_doc(self, fixed = False):
        return self.future_d_list
    def get_satisfaction(self, question_id, doc_id, eeg, anno = 0):
        if self.eeg_classifier != None:
            re = self.eeg_classifier.add_data(question_id, doc_id, eeg, anno) 
    

class Re_ranker():
    def __init__(self, mode = 'eeg', mode_type = 1):
        self.mode = mode
        self.m = 20
        self.eeg_mean = 0.4482
        self.eeg_std = 0.1123
        self.anno = json.load(open('../system/data/total_info_no_filtered.json'))
        # q ddd bset_para
        self.best_pa = json.load(open('../system/data/q2subset2para.json'))
        self.q2d2score = json.load(open(f'../system/data/q2d2score_{self.m}.json'))
        self.q2d2d2score = json.load(open(f'../system/data/q2d2d2score_{self.m}.json'))
        if mode == 'eeg':
            self.eeg_classifier = Eeg_classifier()
        elif mode == 'simulation':
            self.eeg_classifier = Simulator_classifier()
        # 备用的分类器
        self.simple_classifier = Simulator_classifier()
        self.simple_classifier.shot = 500
        self.mode_type = mode_type
        self.offline_alpha = 0.9
        self.offline_click = 1.0
        self.offline_bs = 1.0

    def init_docs(self,q,d_list,intent,):
        self.q = q
        self.d_list = d_list
        self.d2score = {}
        self.now_d_list = []
        self.future_d_list = d_list
        self.intent = intent
        self.future_d_list = [[d, self.q2d2score[q][d]['score']] for d in self.future_d_list]
        self.future_d_list = sorted(self.future_d_list, key= lambda v:v[1], reverse = True)
        self.future_d_list = [d[0] for d in self.future_d_list]
        self.d2info = {}
    
    # click: 1代表点击(land) 0代表非点击(serp)
    # eeg: None代表没信号，否则给出信号
    def update_status(self, question_id, pos, d, click, eeg, anno = 0):
        if d not in self.now_d_list:
            self.now_d_list.append(d)
            self.d2info[d] = {'serp':-1,'land':-1}
            del self.future_d_list[self.future_d_list.index(d)]
            info_list = [0, 0, 0, 0]
            # eeg, click, click(land也算), eeg(land)
        else:
            info_list = self.d2score[d]
        # serp
        if type(eeg) != type(None):
            info_list[0] = self.get_satisfaction(question_id, pos, eeg, anno)
        self.d2score[d] = info_list
    
    def update_land(self, question_id, pos, d, eeg, anno = 0):
        if d not in self.now_d_list:
            self.now_d_list.append(d)
            self.d2info[d] = {'serp':-1,'land':-1}
            del self.future_d_list[self.future_d_list.index(d)]
            info_list = [0, 0, 0, 0]
            # eeg, click, click(land也算), eeg(land)
        else:
            info_list = self.d2score[d]
        if pos != -1:
            info_list[1] = 1
            info_list[2] = 1
        if pos == -2:
            info_list[1] = 0 # sometimes 1?
            info_list[2] = 1
        self.d2info[d]['land'] = anno
        if type(eeg) != type(None):
            info_list[3] = self.get_satisfaction(question_id, pos, eeg, 1 if anno > 2 else 0, land = True)
        self.d2score[d] = info_list

    def update_labels(self, question_id, y, testing = False):
        for i, d in enumerate(self.now_d_list):
            self.d2info[d]['serp'] = y[i]
        self.eeg_classifier.add_label(question_id, y)

    def get_next_doc(self, fixed = False):
        if self.mode_type == 0:
            return self.future_d_list
        if len(self.now_d_list) == 0:
            return self.future_d_list
        if fixed == False and len(self.now_d_list) <= 3:
            try:
                self.now_d_list = sorted(self.now_d_list)
                para = self.best_pa[self.q]
                for d in self.now_d_list:
                    para = para[d]
                alpha, click_gamma, eeg_gamma = para['para']
            except:
                print('para bug')
                alpha, click_gamma, eeg_gamma = 1.5, 1.5, 1.5
                inputs = ''
                while inputs != 'continue':
                    try:
                        print(eval(inputs))
                    except Exception as e:
                        print(e)
                    inputs = input()
        else:
            alpha, click_gamma, eeg_gamma = 1.5, 1.5, 1.5
        future_score = self.bert_qm_all(alpha, click_gamma, eeg_gamma)
        future_score_doc = [[future_score[i], self.future_d_list[i]] for i in range(len(future_score))]
        future_score_doc = sorted(future_score_doc, key = lambda v : v[0], reverse = True)
        return [item[1] for item in future_score_doc]

    def offline_ranking(self, mode):
        # mode bert-click; bert-bs+click
        for d in self.d2score.keys():
            if self.d2score[d][3] != 0:
                self.d2score[d][0] = self.d2score[d][3]
        if mode == 'bert-bs+click':
            self.now_d_score = [[d, self.offline_alpha * (self.d2score[d][0] * self.offline_bs + self.d2score[d][1] * self.offline_click) + (1 - self.offline_alpha) * self.q2d2score[self.q][d]['score']] for d in self.now_d_list]
        elif mode == 'bert-click':
            self.now_d_score = [[d, self.offline_alpha * (self.d2score[d][0] * 0 + self.d2score[d][2] * self.offline_click) + (1 - self.offline_alpha) * self.q2d2score[self.q][d]['score']] for d in self.now_d_list]
        self.now_d_score = sorted(self.now_d_score, key = lambda v: v[1], reverse=True)
        return [item[0] for item in self.now_d_score]

    def get_satisfaction(self, question_id, doc_id, eeg, anno = 0, land = False):
        if random.random() > 0.0:
            if type(eeg) != type(None):
                re = self.eeg_classifier.add_data(question_id, doc_id, eeg, anno, land) 
                if type(re) != type(None):
                    try:
                        re = float(re)
                        return re
                    except:
                        print('re', re)
                        inputs = ''
                        while inputs != 'continue':
                            try:
                                print(eval(inputs))
                            except Exception as e:
                                print(e)
                            inputs = input()
        tmp_mean = np.mean(self.eeg_classifier.y_pred) if len(self.eeg_classifier.y_pred) > 20 else 0.4482
        self.simple_classifier.para['anno'][0]['loc'] = tmp_mean - 0.3
        self.simple_classifier.para['anno'][1]['loc'] = tmp_mean + 0.5
        return self.simple_classifier.add_data(question_id, doc_id, eeg, anno)

    def bert_qm_init(self, q2d2score, q, now_d_list, future_d_list, d2score, click_gamma, eeg_gamma, kd = 10, kc = 10):
        now_d_list_bert_score = [[d,q2d2score[q][d]['score']] for d in now_d_list]
        if type(d2score) == type(None):
            now_d_list_bert_score_select = sorted([[now_d_list_bert_score[i][0], now_d_list_bert_score[i][1]] for i in range(len(now_d_list_bert_score))], key = lambda v: v[1], reverse = True)
        else:
            now_d_list_bert_score_select = sorted([[item[0], item[1] + eeg_gamma * d2score[item[0]][0] + click_gamma * d2score[item[0]][1]] for i, item in enumerate(now_d_list_bert_score)], key=lambda v: v[1], reverse=True)

        future_score1 = [q2d2score[q][d]['score'] for d in future_d_list]
        now_d_list_bert_score_select = now_d_list_bert_score_select[:kd]
        now_d_list_set = set([item[0] for item in now_d_list_bert_score_select])
        now_d_list_bert_split_score = [[d,pos,score] for d in now_d_list for pos,score in q2d2score[q][d]['split_score'].items() if d in now_d_list_set]
        if type(d2score) == type(None):
            pass
        else:
            now_d_list_bert_split_score = [[now_d_list_bert_split_score[i][0], now_d_list_bert_split_score[i][1], now_d_list_bert_split_score[i][2] + click_gamma * d2score[now_d_list_bert_split_score[i][0]][1] + eeg_gamma * d2score[now_d_list_bert_split_score[i][0]][0]] for i in range(len(now_d_list_bert_split_score))]
        try:
            now_d_list_bert_split_score = sorted(now_d_list_bert_split_score, key = lambda v: v[2], reverse = True)[:kc]
        except:
            print('now_d_list_bert_split_score bug-----------')
            inputs = ''
            while inputs != 'continue':
                try:
                    print(eval(inputs))
                except Exception as e:
                    print(e)
                inputs = input()
        return future_score1, now_d_list_bert_split_score


    def bert_qm_all(self, alpha, click_gamma, eeg_gamma):
        future_score1, now_d_list_bert_split_score = self.bert_qm_init(self.q2d2score, self.q, self.now_d_list, self.future_d_list, self.d2score, click_gamma, eeg_gamma)
        future_score2 = bert_qm_procedure(self.q, now_d_list_bert_split_score, self.future_d_list, self.q2d2d2score,)
        future_score = combine_array(future_score2, future_score1, alpha)
        return future_score

# unit test
if __name__ == '__main__':
    # teset Eeg_classifier
    # eeg_classifier = Eeg_classifier()
    # with open('/home/yzy/online/data/raw/merged/1_processed.json') as f:
    #     data = json.load(f)
    # count = 0
    # y = []
    # for i in range(len(data)):
    #     if data[i]['motion'] != 'serp':
    #         continue
    #     eeg_classifier.add_data(data[i]['raw'])
    #     y.append(data[i]['score'])
    #     count += 1
    #     if count == 10:
    #         eeg_classifier.add_label(y)
    #         count = 0
    #         y = []
    # print(eeg_classifier.get_performance())

    # test Re_ranker

    q2intent2pos = {}
    u2info = json.load(open('/home/yzy/online/data/preprocessed/u2info.0828.json'))
    for u in u2info.keys():
        for raw_q in u2info[u][0].keys():
            q = u2info[u][1][raw_q]['q']
            intent = u2info[u][1][raw_q]['intent']
            raw_d = list(u2info[u][0][raw_q].keys())[-1]
            pos = len(u2info[u][0][raw_q][raw_d]['now_d_list'])
            if q not in q2intent2pos.keys():
                q2intent2pos[q] = {}
            if intent not in q2intent2pos[q].keys():
                q2intent2pos[q][intent] = []
            q2intent2pos[q][intent].append(pos)
    for q in q2intent2pos.keys():
        for intent in q2intent2pos[q].keys():
            q2intent2pos[q][intent] = int(np.mean(q2intent2pos[q][intent]))

    def random_data_test(ranker, displayer, random_data):
        random.seed(2022)
        final_pos_list = [random.randint(6,15) for i in range(len(random_data))]
        displayer.q2d2score = ranker.q2d2score

        for i, simulation in enumerate(random_data):
            q = simulation['q']
            d_list = [item['did'] for item in simulation['doc_list']]
            intent_list = [key for key in simulation['intent'].keys() if simulation['intent'][key] != '']
            intent = intent_list[0]
            ranker.init_docs(q, d_list, intent)
            
            d2anno = dict([[item['did'], int(intent) in item['anno']] for item in simulation['doc_list']])
            displayer.add_true(q, d2anno)

            if q in q2intent2pos.keys() and intent in q2intent2pos[q].keys():
                final_pos = q2intent2pos[q][intent]
            else:
                final_pos = final_pos_list[i]
            final_pos = min(len(d_list) - 1, final_pos)

            labels = []
            prev_d = []
            for pos in range(0, final_pos):
                ds = ranker.get_next_doc(fixed = True)
                d = ds[0]
                labels.append(4 if d2anno[d] == 1 else 1)
                # pos, d, click, eeg
                click = 0
                
                if pos == 3:
                    for i in range(0,3):
                        if d2anno[prev_d[i]] == 1:
                            ranker.update_status(q, pos, prev_d[i], 1, eeg=None, anno = d2anno[d])
                if pos < 3:
                    prev_d.append(d)
                elif d2anno[d] == 1:
                    click = 1
                
                ranker.update_status(q, pos, d, click, eeg=None, anno = d2anno[d])
                
                if pos > 0:
                    displayer.add_info(q, ds)

            ranker.update_labels(q, labels)
        displayer.show()
        return displayer

    for i in range(4,20):
        re_ranker = Re_ranker(mode='simulation')
        simple_ranker = Simple_ranker()
        with open(f'../../random_data/{i}.json') as f:
            random_data = json.load(f)
        displayer2 = random_data_test(simple_ranker, Displayer(), random_data)
        displayer1 = random_data_test(re_ranker, Displayer(), random_data)

        import scipy
        for key in displayer1.offline_list.keys():
            if key != 'q':
                print(f'{np.mean(displayer1.offline_list[key])}')
        for key in displayer2.offline_list.keys():
            if key != 'q':
                print(f'{np.mean(displayer2.offline_list[key])}')
        print(scipy.stats.ttest_ind(displayer1.offline_list['ndcg@5'], displayer2.offline_list['ndcg@5']))
    
    # test reranker using u2info.json
    # import tqdm
    
    # anno = json.load(open('/home/yzy/online/text_data/merged_data/total_info_no_filtered.json'))
    # u2info = json.load(open('/home/yzy/online/data/preprocessed/u2info.0828.json'))

    # def test_u2info(u2info, anno, Ranker, ):
    #     offline_list = {}
    #     for u in tqdm.tqdm(u2info.keys()):
    #         displayer = Displayer()
    #         simple_ranker = Ranker(mode='simulation')
    #         for raw_q in u2info[u][0].keys():
    #             q = u2info[u][1][raw_q]['q']
    #             d_list = list(anno[q].keys())
    #             anno_list = [anno[q][d]['anno'] for d in d_list]
    #             intent = u2info[u][1][raw_q]['intent']
    #             simple_ranker.init_docs(q, d_list, intent)
    #             raw_d = list(u2info[u][0][raw_q].keys())[-1]
    #             now_d_list = u2info[u][0][raw_q][raw_d]['now_d_list']
    #             d2anno = dict([[d, int(intent) in anno_list[d_i]] for d_i, d in enumerate(d_list)])
    #             displayer.add_true(q, d2anno)

    #             d2score = {}
    #             for item in u2info[u][2][raw_q]:
    #                 if item['d'] not in d2score.keys():
    #                     d2score[item['d']] = item['score']
    #                 if item['motion'] == 'serp':
    #                     d2score[item['d']] = item['score']
    #             labels = []

    #             for pos, d in enumerate(now_d_list):      
    #                 # if pos < 3:
    #                 #     click = 0
    #                 # el
    #                 if pos < 3:
    #                     click = 0
    #                 elif pos == 3:
    #                     for i in range(0,3):
    #                         if d2anno[now_d_list[i]] == 1:
    #                             simple_ranker.update_status(pos, d, 1, eeg=None)
    #                 if d2anno[d] == 1:
    #                     click = 1   
    #                 else:
    #                     click = 0   
    #                 simple_ranker.update_status(pos, d, click, eeg=d2anno[d])
    #                 ds = simple_ranker.get_next_doc(fixed = True)
    #                 displayer.add_info(q, ds,)
    #                 labels.append(d2score[d])

    #             simple_ranker.update_labels(labels)
    #         offline_list[u] = displayer.offline_list
    #     return offline_list

    # offline_list1 = test_u2info(u2info, anno, Simple_ranker, )
    # offline_list2 = test_u2info(u2info, anno, Re_ranker, )

    # import scipy
    # for u in offline_list1.keys():
    #     for key in offline_list1[u].keys():
    #         print(key, np.mean(offline_list2[u][key]) - np.mean(offline_list1[u][key]), end = '\t')
    #     print(scipy.stats.ttest_ind(offline_list2[u]['map'], offline_list1[u]['map']))




import threading
from sklearn.metrics import ndcg_score, average_precision_score
import numpy as np
import json

def add_result(y_true, y_pred, result_dic, q):
    if np.std(y_true) == 0:
        return
    for k in [1,3,5]:
        result_dic[f'ndcg@{k}'].append(ndcg_score([y_true],[y_pred],k=k))
    result_dic['map'].append(average_precision_score(y_true, y_pred))
    result_dic['q'].append(q)

class Displayer():
    def __init__(self):
        self.user_id = None
        self.q2d2score = None
        self.q2d2anno = {}
        self.offline_list = {'ndcg@1':[], 'ndcg@3':[], 'ndcg@5':[],'map':[], 'q':[]}
        self.offline_relative_list = {'ndcg@1':[], 'ndcg@3':[], 'ndcg@5':[],'map':[], 'q':[]}
        self.online_list = {'session_length':[], 'satisfaction':[], 'difficulity':[], 'q':[]}
        self.q2ds = {}
        self.evaluator = None
        self.trigger_lost = 0
        self.f = open('tmp.txt','w')
        self.dump_data_thread = None
        self.data = {'y_true':[], 'y_pred':[], 'q':[], 'ds':[] }

    def add_info(self,q,ds,):
        if q not in self.q2ds.keys():
            self.q2ds[q] = []
        self.q2ds[q].append(ds)
        self.evaluate(q,ds)
        
       
    def add_online_info(self, q, session_length, satisfaction, difficulity):
        self.online_list['session_length'].append(session_length)
        self.online_list['satisfaction'].append(satisfaction)
        self.online_list['difficulity'].append(difficulity)
        self.online_list['q'].append(q)

    def add_true(self,q, d2score):
        self.q2d2anno[q] = d2score     

    def evaluate(self,q,ds,):
        
        y_pred = [-i for i in range(len(ds))]
        y_true = [self.q2d2anno[q][d] for d in ds]
        add_result(y_true, y_pred, self.offline_list,q)
        y_pred = [self.q2d2score[q][d]['score'] for d in ds]
        add_result(y_true, y_pred, self.offline_relative_list,q)
        

        self.data['y_true'].append(y_true)
        self.data['y_pred'].append(y_pred)
        self.data['q'].append(q)
        self.data['ds'].append(ds)

    def dump_data(self, ):
        pass

    # 输出评价结果
    def show(self,):
        pass

        




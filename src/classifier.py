from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, ndcg_score, roc_auc_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import numpy as np

class Base_models:
    def __init__(self, classifier, args, truth_label = 1):
        self.classifier_name = classifier
        if 'warm_start' not in args:
            args['warm_start'] = False
        if classifier == 'lr':
            if 'C' in args.keys():
                self.classifier = LogisticRegression(C=args['C'])
            else:
                self.classifier = LogisticRegression(C=0.001)
        elif classifier == 'gbdt':
            self.classifier = GradientBoostingClassifier(random_state=2020,
                               learning_rate=args['lr'], n_estimators=args['epoches'],
                               max_depth=args['depth'], min_samples_split=args['split'],
                               min_samples_leaf=args['leaf'],max_features=args['features'],
                                subsample=args['subsample'],verbose=args["verbose"],warm_start=args['warm_start'])
        elif classifier == 'mlp': # is wrong now
            self.classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', 
                        beta_1=0.9, beta_2=0.999, early_stopping=False,
                        epsilon=1e-08, hidden_layer_sizes=(15, ), learning_rate='adaptive',
                        learning_rate_init=0.1, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)
        elif classifier == 'svm':
            # self.classifier = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf', max_iter = 4000, class_weight='balanced')
            self.classifier = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf', max_iter = 4000)
        elif classifier == 'random':
            self.classifier = None

        self.truth_label = truth_label
        self.mean_score = 0.5
        self.eeg_mean = 0.4482
        self.is_fit = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if self.classifier != None:
            self.classifier.fit(X, y)
            self.mean_score = np.mean(y)
        self.is_fit = True
    
    def predict(self, X):
        if self.classifier == None:
            prob = [[np.random.random(), 0] for i in range(len(X))]
            prob = [[prob[i][0], 1 - prob[i][0]] for i in range(len(X))] 
            pred = [1 if i[1] > 0.5 else 0 for i in prob]
        else:
            pred = self.classifier.predict(X).tolist()
        return pred

    def predict_proba(self, X, normalized= True):
        if self.classifier == None:
            prob = [[np.random.random(), 0] for i in range(len(X))]
        elif hasattr(self, 'is_fit') and self.is_fit == False:
            prob = [self.eeg_mean for i in range(len(X))]
        else:
            prob = [item[1] for item in self.classifier.predict_proba(X).tolist()]
        if normalized and hasattr(self, 'mean_score'):
            return [item - self.mean_score + 0.5 for item in prob]
        else:
            return prob
        
    def classifier_evaluate(self, X, y):
        pred, prob = self.predict(X)
        acc = accuracy_score(y, pred)
        precision = precision_score(y, pred, pos_label = self.truth_label)
        recall = recall_score(y, pred, pos_label = self.truth_label)
        f1 = f1_score(y, pred, pos_label = self.truth_label, average='binary')
        auc = roc_auc_score(y, [prob[i][1] for i in range(len(prob))],)

        return acc, precision, recall, f1, auc
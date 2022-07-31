import numpy as np
from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode 
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold

class BaggingClassifier2(BaseEnsemble, ClassifierMixin):
    """Bagging Classifier"""
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True, weight_mode=False):
        #Klasyfikator bazowy
        self.base_estimator = base_estimator
        #Liczba Klasyfikatorów
        self.n_estimators = n_estimators
        #Czy na podstawie głosów większościowych
        self.hard_voting = hard_voting
        #Wagi
        self.weight_mode = weight_mode
        #Ustawienie ziarna losowości
        self.random_state = random_state
        np.random.seed(self.random_state)
    
    def fit(self, X, y):
        """Trening"""
        #Sprawdzenie czy X i y maja własciwy kształt
        X, y = check_X_y(X,y)
        #Przechowywanie nazw klas
        self.classes_ = np.unique(y)
        #Zapis liczby atrybutów
        self.n_features = X.shape[1]
        self.weights = np.zeros(self.n_estimators)
        
        #macierz na wyniki
        self.ensemble_ = []
        #Bagging
        for i in range(self.n_estimators):
            self.bootstrap = np.random.choice(len(X),size=len(X), replace=True)
            self.ensemble_.append(clone(self.base_estimator).fit(X[self.bootstrap], y[self.bootstrap]))
            self.weights[i] = accuracy_score(self.ensemble_[i].predict(X), y)
        return self
    
    def predict(self, X):
        #sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        X = check_array(X)

        #głosowanie większościowe
        if self.hard_voting:
            pred_ = []
            
            if self.weight_mode == True:
                #option number one:
                """
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))
                    pred_[i] = pred_[i]*self.weights[i]
                pred_ = np.array(pred_)
                mean_pred_ = np.mean(pred_, axis=0)
                prediction = np.around(mean_pred_).astype(int)
                return self.classes_[prediction]
                """
                #option number two:
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))
                    pred_[i] = pred_[i]*self.weights[i]
                
                pred_ = np.array(pred_)
                prediction = mode(pred_, axis=0)[0].flatten()
                prediction = np.around(prediction).astype(int)
                return self.classes_[prediction]
            
            else:
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))

                pred_ = np.array(pred_)
                prediction = mode(pred_, axis=0)[0].flatten()
                return self.classes_[prediction]
        #akumulacja wsparć
        else:
            if self.weight_mode == False:
                esm = self.ensemble_support_matrix(X)
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=1)
                return self.classes_[prediction]

            else:
                esm = self.ensemble_support_matrix(X)
                for i, clf in enumerate(self.ensemble_):
                    esm[i] = esm[i] * self.weights[i]
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=1)
                return self.classes_[prediction]
                  
    def ensemble_support_matrix(self, X):
        """Macierz wsparć"""
        probas_ = []
        
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X))

        return np.array(probas_)
if __name__ == '__main__':
    datasets = ['banana']

    clfs = {
        'Bagging NHV W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=True, random_state=1234),
        'Bagging NHV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=False, random_state=1234),
        'Bagging HV W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=True, random_state=1234),
        'Bagging HV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=False, random_state=1234),
    }   

    n_repeat = 5
    n_split = 2
    scores = np.zeros((len(clfs), len(datasets), n_split*n_repeat))
    rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

    #for data_id, dataset in enumerate(datasets):
        #dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
        #X = dataset[:, :-1]
        #y = dataset[:, -1].astype(int)
    X, y = make_classification(
            n_samples=100, n_classes=4, n_informative=4, random_state=100)
    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, 0, fold_id] = accuracy_score(y[test], y_pred)

    mean = np.mean(scores, axis=2)
    std = np.std(scores, axis=2)
    for clf_id, clf_name in enumerate(clfs):
        print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))
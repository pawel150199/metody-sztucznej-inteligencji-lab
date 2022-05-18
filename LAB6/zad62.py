
import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone 
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode 

class BaggingClassifier2(BaseEnsemble, ClassifierMixin):
    """Bagging Classifier"""
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True, weights=False):
        #Klasyfikator bazowy
        self.base_estimator = base_estimator
        #Liczba Klasyfikatorów
        self.n_estimators = n_estimators
        #Czy na podstawie głosów większościowych
        self.hard_voting = hard_voting
        #Wagi
        self.weights = weights
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
        self.weights = []
        if self.weights == True:
            n_split = 5
            n_repeat = 10
            scores  = np.zeros((self.n_estimators, n_split*n_repeat))
            for i in range(self.n_estimators):
                clf = clone(self.base_estimator)
                clf.fit(X, y)
                y_pred = clf.predict(X)
                scores[i] = accuracy_score(y, y_pred)
            self.weights = scores*y_pred
                    
        else:
            #macierz na wyniki
            self.ensemble_ = []
            #Bagging
            for i in range(self.n_estimators):
                self.bootstrap = np.random.choice(len(X),size=len(X), replace=True)
                self.ensemble_.append(clone(self.base_estimator).fit(X[self.bootstrap], y[self.bootstrap]))

        return self
    
    def predict(self, X):
        #sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        X = check_array(X)

        #głosowanie większościowe
        if self.hard_voting:
            pred_ = []
            
            if self.weights == True:
                pass
                """
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))

                pred_ = np.array(pred_)
                temp = []
                scores = np.zeros((pred_.T.shape[0], np.sum(self.ranks) + self.n_estimators))

                for k in range(0, pred_.T.shape[0]):
                    for i in range(0, self.n_estimators):
                        for j in range(0, self.ranks[i]):
                            temp.append(int(pred_.T[k][i]))

                    scores[k] = np.append(pred_.T[k], temp)
                    temp = []

                scores = scores.astype(int)
                prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=scores)

                return self.classes_[prediction]
                """
            #do akumulacji wsparc
            else:
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))

                pred_ = np.array(pred_)
                prediction = mode(pred_, axis=0)[0].flatten()

                return self.classes_[prediction]

        else:
            if self.weights == False:
                esm = self.ensemble_support_matrix(X)
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=0)
                return self.classes_[prediction]

            else:
                average_support = self.ensemble_support_matrix(X)
                prediction = np.argmax(average_support, axis=0)

                return self.classes_[prediction]
    def ensemble_support_matrix(self, X):
        """Macierz wsparć"""
        probas_ = []
        
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.bootstrap[i]]))

        return np.array(probas_)





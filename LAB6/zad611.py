import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from numpy.random import randint
from scipy.stats import mode
from sklearn.base import ClassifierMixin, clone

ENSAMBLE_SIZE=5

class OwnBaggingClasifier(BaseEnsemble,ClassifierMixin):

    def init(self, base_estimator=DecisionTreeClassifier(random_state=4413),n_subspace_features=5, n_estimators=5, hard_voting=True, random_state=4413, scales=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        self.n_subspace_features = n_subspace_features
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)


    def fit(self,X,y):
        self.clfs = []

        if self.hard_voting == True:
            for i in range(ENSAMBLE_SIZE):
                clf = clone(DecisionTreeClassifier())
                bootstrap = np.random.choice(len(X),size=len(X), replace=True)
                clf.fit(X[bootstrap],y[bootstrap])
                self.clfs.append(clf)

            return self
        else:
            esm = self.ensemble_support_matrix(self,X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
            return self.classes_[prediction]

    def predict(self,X):
        predictions = []

        for clf in self.clfs:
            predictions.append(clf.predict(X))

        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()

    def ensemble_support_matrix(self,X):

        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba)
        return probas_
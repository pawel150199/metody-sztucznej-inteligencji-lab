from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin, BaseEstimator



class SamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, base_preprocesing=None):
        self.base_estimator = base_estimator
        self.base_preprocesing = base_preprocesing
    
    def fit(self, X, y):
        if self.base_preprocesing != None:
            preproc = clone(self.base_preprocesing)
            X_new, y_new = preproc.fit_resample(X, y)
            self.clf = clone(self.base_estimator)
            self.clf.fit(X_new, y_new)
            return self
        else:
            self.clf = clone(self.base_estimator)
            self.clf.fit(X, y)
            return self

    def predict(self, X):
        prediction = self.clf.predict(X)
        return prediction

if __name__ == '__main__':
    X, y = datasets.make_classification(
    n_samples=100, n_classes=2, n_informative=2, random_state=100)


    n_splits = 2
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

    clf = SamplingClassifier(base_estimator=DecisionTreeClassifier(random_state=123), base_preprocesing=RandomUnderSampler(random_state=123))
    scores = []
    for train, test in rskf.split(X, y):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        # print(accuracy_score(y[test], y_pred))
        scores.append(accuracy_score(y[test], y_pred))
    print("accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))


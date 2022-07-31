import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.base import clone
from scipy.stats import mode

BASE_MODEL = DecisionTreeClassifier()
ENSAMBLE_SIZE = 5


class BaggingClf(ClassifierMixin, BaseEstimator):
    def __init__(self, hard_voting=True, weighted=False):
        self.hard_voting = True
        self.weighted = weighted
        self.weights = None
        self.clfs_ = None

    def fit(self, X, y):
        self.clfs_ = []
        self.weights = np.zeros(ENSAMBLE_SIZE)

        for i in range(ENSAMBLE_SIZE):
            clf = clone(BASE_MODEL)
            bootstrap = np.random.choice(len(X), size=len(X), replace=True)
            clf.fit(X[bootstrap], y[bootstrap])
            self.weights[i] = accuracy_score(clf.predict(X), y)
            self.clfs_.append(clf)

        return self

    def predict(self, X):
        predictions = []
        if self.weighted:
            for clf in self.clfs_:
                predictions.append(clf.predict(X))
            predictions = np.array(predictions)
            pred = np.zeros(predictions.T.shape[0], dtype=int)
            for idx, row in enumerate(predictions.T):
                w = row*self.weights
                if np.sum(w) >= ENSAMBLE_SIZE/2:
                    pred[idx] = 1
                else:
                    pred[idx] = 0
                pred[idx] = int(pred[idx])
            return pred

        else:
            for clf in self.clfs_:
                predictions.append(clf.predict(X))
            predictions = np.array(predictions)
            return mode(predictions, axis=0)[0].flatten()


X, y = datasets.make_classification(
    n_samples=100, n_classes=2, n_informative=2, random_state=100)


n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

clf = BaggingClf(weighted=True)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    # print(accuracy_score(y[test], y_pred))
    scores.append(accuracy_score(y[test], y_pred))
print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

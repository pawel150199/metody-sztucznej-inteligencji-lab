import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
BASE_MODEL = DecisionTreeClassifier()
ENSAMBLE_SIZE = 5

class BaggingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hard_voting=True, weights=False):
        self.hard_voting = True
        self.weights = weights
        self._clfs = None

    def fit(self, X, y):
        self._clfs = []
        for _ in range(ENSAMBLE_SIZE):
            clf = clone(BASE_MODEL)
            bootstrap = np.random.choice(len(X), size=len(X), replace=True)
            clf.fit(X[bootstrap], y[bootstrap])
            self._clfs.append(clf)

            return self
    def predict(self, X):
        predictions = []

        for clf in self._clfs:
            predictions.append(clf.predict(X))

        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()
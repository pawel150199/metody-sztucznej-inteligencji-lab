from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

class BaggingEnsemble(BaggingClassifier, ClassifierMixin):
    def __init__(self, random_state=None):
        self.base_estimator = DecisionTreeClassifier()
        self.estimators = 5
        self.n_subspace_features = 5
        self.random_state = random_state
        np.random.seed(self.random_state)
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.subspaces = np.random.randint(0, self.n_features, (self.estimators, self.n_subspace_features))
        self.ensemble = []
        for i in range(self.estimators):
            self.ensemble.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))
        return self

    def predict(self, X):
        predict_temp = []
        for i, clf in enumerate(self.ensemble):
            predict_temp.append(clf.predict(X[:, self.subspaces[i]]))
        predict_temp = np.array(predict_temp)
        class_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predict_temp.T)
        return self.classes[class_pred]
if __name__ == "__main__":
    dataset = 'yeast4'
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1337)
    bagging = BaggingEnsemble(random_state=1337)
    cart = DecisionTreeClassifier(random_state=1337)
    bagging_scores = []
    cart_scores = []

    for train, test in rskf.split(X, y):
        bagging.fit(X[train], y[train])
        cart.fit(X[train], y[train])
        y_bagging = bagging.predict(X[test])
        y_cart = cart.predict(X[test])
        bagging_scores.append(accuracy_score(y[test], y_bagging))
        cart_scores.append(accuracy_score(y[test], y_cart))
    print("BAGGING: %.3f (%.3f)" % (np.mean(bagging_scores), np.std(bagging_scores)))
    print("CART: %.3f (%.3f)" % (np.mean(cart_scores), np.std(cart_scores)))
import numpy as np
import enum
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

class myBagEns(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator = DecisionTreeClassifier(), n_estimators = 5, p_bag_samples = 0.8, hard_voting = True, weights_pred = False, random_state = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.p_bag_samples = p_bag_samples
        self.hard_voting = hard_voting
        self.random_state = random_state
        self.weights_pred = weights_pred

        np.random.seed(self.random_state)

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        n_bag_samples = int((X.shape[0])*self.p_bag_samples)

        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        if n_bag_samples > self.n_samples:
            raise ValueError("Number of samples in a bag is be higher than the number of samples.")

        self.bags = np.random.randint(0, self.n_samples, (self.n_estimators, n_bag_samples))
        self.ensamble_ = []
        self.weights = np.zeros(self.n_estimators)

        for i in range(self.n_estimators):
            self.ensamble_.append(clone(self.base_estimator).fit(X[self.bags[i]], y[self.bags[i]]))
            self.weights[i] = accuracy_score(self.ensamble_[i].predict(X), y)
        
        return self

    def predict(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)

        if self.hard_voting:
            if self.weights_pred:                                           #voting = true, weights = true
                pred_ = []
            
                for i, member_clf in enumerate(self.ensamble_):
                    pred_.append(member_clf.predict(X))
                
                pred_ = np.array(pred_)

                prediction = np.zeros(pred_.T.shape[0], dtype=int)
                for i, row in enumerate(pred_.T):
                    t = row*self.weights
                    if np.sum(t) >= (self.n_estimators/2):
                        prediction[i] = 1
                    else:
                        prediction[i] = 0
                    prediction[i] = int(prediction[i])
                
                return self.classes_[prediction]
            else:                                                           #voting = true, weights = false
                pred_ = []
                
                for i, member_clf in enumerate(self.ensamble_):
                    pred_.append(member_clf.predict(X))
                
                pred_ = np.array(pred_)
                
                prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
                
                return self.classes_[prediction]
        else:
            if self.weights_pred:                                           #voting = false, weights = true
                self.probas = np.zeros((self.n_estimators, X.shape[0], self.classes_.shape[0]))
                
                for i, clf in enumerate(self.ensamble_):
                    self.probas[i] = clf.predict_proba(X)
                    self.probas[i] = self.probas[i]*self.weights[i]

                average_support = np.mean(self.probas, axis=0)

                prediction = np.argmax(average_support, axis=1)
                
                return self.classes_[prediction]
            else:                                                           #voting = false, weights = false
                self.probas = np.zeros((self.n_estimators, X.shape[0], self.classes_.shape[0]))

                for i, clf in enumerate(self.ensamble_):
                    self.probas[i] = clf.predict_proba(X)

                average_support = np.mean(self.probas, axis=0)
                
                prediction = np.argmax(average_support, axis=1)
                
                return self.classes_[prediction]

dataset = np.genfromtxt(f"datasets/australian.csv", delimiter = ',')
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
#X, y = datasets.make_classification(n_samples=200, n_features=10, n_informative=8, n_redundant = 0, n_repeated=2, random_state=1234)

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1254)

clf = myBagEns(random_state=1253)
tree = DecisionTreeClassifier(random_state=1253)
scores_b = []
scores_t = []

for train, test in rskf.split(X, y):

    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores_b.append(accuracy_score(y[test], y_pred))

    tree.fit(X[train], y[train])
    y_pred = tree.predict(X[test])
    scores_t.append(accuracy_score(y[test], y_pred))

print("Accuracy score - bagging: %.3f (%.3f)" % (np.mean(scores_b), np.std(scores_b)))
print("Accuracy score - tree: %.3f (%.3f)" % (np.mean(scores_t), np.std(scores_t)))
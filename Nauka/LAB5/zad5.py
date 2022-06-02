# --- --- --- --- --- --- --- --- --- zad 5.1 --- --- --- --- --- --- --- --- --- --- --- --- --- 

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

dataset = 'yeast4'
dataset = np.genfromtxt("datasets/              §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§%s.csv" % (dataset), delimiter=",")

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

print('--- --- --- --- --- --- --- --- --- zad 5.1 --- --- --- --- --- --- --- --- --- --- --- --- ---')
print("BAGGING: %.3f (%.3f)" % (np.mean(bagging_scores), np.std(bagging_scores)))
print("CART: %.3f (%.3f)" % (np.mean(cart_scores), np.std(cart_scores)))

# --- --- --- --- --- --- --- --- --- zad 5.2 --- --- --- --- --- --- --- --- --- --- --- --- --- 

print('--- --- --- --- --- --- --- --- --- zad 5.2 --- --- --- --- --- --- --- --- --- --- --- --- ---')
class BaggingEnsemble2(BaggingClassifier, ClassifierMixin):
    def __init__(self, hard_voting, weight, random_state=None):
        self.base_estimator = DecisionTreeClassifier()
        self.estimators = 5
        self.n_subspace_features = 5
        self.hard_voting = hard_voting
        self.weight = weight
        self.random_state = random_state
        np.random.seed(self.random_state)
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        self.subspaces = np.random.randint(0, self.n_features, (self.estimators, self.n_subspace_features))
        self.ensemble = []
        for i in range(self.estimators):
            self.ensemble.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))
        self.accuary_sum = []
        for i, member_clf in enumerate(self.ensemble):
            self.accuary_sum.append(accuracy_score(y, member_clf.predict(X[:, self.subspaces[i]])))
            self.accuary_sum[i] = np.mean(self.accuary_sum[i])
        return self

    def predict(self, X):
        if self.hard_voting and self.weight:
            predict_temp = []
            for i, clf in enumerate(self.ensemble):
                predict_temp.append(clf.predict(X[:, self.subspaces[i]]))
            predict_temp = np.array(predict_temp)
            class_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predict_temp.T)
        elif not self.hard_voting and not self.weight:
            proba = []
            for i, clf in enumerate(self.ensemble):
                proba.append(clf.predict_proba(X[:, self.subspaces[i]]))
            avg = np.mean(proba, axis=0)
            class_pred = np.argmax(avg, axis=1)

        return self.classes[class_pred]

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'vowel0', 'waveform', 'wisconsin', 'yeast3']

clfs = {
    'bagging default': BaggingEnsemble(random_state=1337),
    'bagging hard weight true': BaggingEnsemble2(random_state=1337, hard_voting=False, weight=False),
    'bagging hard weight false': BaggingEnsemble2(random_state=1337, hard_voting=True, weight=True),
    'cart': DecisionTreeClassifier(random_state=1337)
}

# Zakomentowana pętla poniżej, ponieważ puściłem ją raz i otrzymałem wyniki :)


rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1337)

scores = np.zeros((len(clfs), 20, 5 * 10))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
mean_scores = np.mean(scores, axis=2).T

from scipy.stats import rankdata
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)

mean_ranks = np.mean(ranks, axis=0)

from scipy.stats import ranksums

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

from tabulate import tabulate

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
#print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
#print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

# --- --- --- --- --- --- --- --- --- zad 5.3 --- --- --- --- --- --- --- --- --- --- --- --- --- 

from sklearn.ensemble import BaseEnsemble

class RandomSubspace(BaseEnsemble, ClassifierMixin):
    def __init__(self, hard_voting=True, random_state=None):
        self.base_estimator = None
        self.n_estimators = 5
        self.n_subspace_features = 5
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))
        self.ensemble = []
        for i in range(self.n_estimators):
            self.ensemble.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))
        return self

    def predict(self, X):
        if self.hard_voting:
            predict_temp = []
            for i, member_clf in enumerate(self.ensemble):
                predict_temp.append(member_clf.predict(X[:, self.subspaces[i]]))
            predict_temp = np.array(predict_temp)
            predict_temp = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predict_temp.T)
        return self.class_pred[predict_temp]



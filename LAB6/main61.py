from statistics import mean
from matplotlib.pyplot import axis
import numpy as np
from zad61 import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from zad62 import BaggingClassifier2
datasets = ['australian']

clfs = {
    'Bagging': BaggingClassifier(random_state=1234, n_estimators=5),
    'CART': DecisionTreeClassifier(random_state=1234),
    "Bagging v2": BaggingClassifier2(random_state=1234, n_estimators=5, hard_voting=True, scales=False)
}

n_repeat = 5
n_split = 2
scores = np.zeros((len(clfs), len(datasets), n_split*n_repeat))
rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=2)
std = np.std(scores, axis=2)
for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))





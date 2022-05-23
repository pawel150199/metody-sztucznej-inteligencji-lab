from zad63  import RandomSubspaceEnsemble
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from zad63 import RandomSubspaceEnsemble
from zad62 import BaggingClassifier2
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

datasets = ['banana', 'balance', 'appendicitis', 'iris']

clfs = {
    'Bagging HV, W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=True, random_state=1234),
    'Bagging HV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=False, random_state=1234),
    'Bagging NHV W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=True, random_state=1234),
    'Bagging NHV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=False, random_state=1234),
    'RSM HV': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=1234), hard_voting=True, random_state=1234, n_subspace_features=2),
    'RSM NHV': RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=1234), hard_voting=False, random_state=1234, n_subspace_features=2)
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
#for clf_id, clf_name in enumerate(clfs):
    #print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))

headers = list(clfs.keys())
headers.insert(0, "dataset\clf")
n_column = np.reshape(np.array(datasets), (len(datasets),1))
mean = np.mean(scores, axis=2).T
std = np.std(scores, axis=2).T

table = np.concatenate((n_column, mean), axis=1)
table = tabulate(table, headers, floatfmt=".2f")
print("Wartości Średnie:\n\n\n", table)

table_2 = np.concatenate((n_column, std), axis=1)
table_2 = tabulate(table_2, headers, floatfmt=".2f")
print("\n\nOdchylenie standardowe:\n\n\n", table_2)
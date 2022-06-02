from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

import numpy as np
from sklearn import datasets
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from zad2 import BaggingClf
from zad3 import RandomSubspaceEnsemble
from scipy.stats import ttest_rel
from tabulate import tabulate

X, y = datasets.make_classification(
    n_samples=100, n_classes=2, n_informative=2, random_state=100)

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

clfs = {
    'Bagging Hard voting = True': BaggingClf(hard_voting=True, weighted=False),
    'Bagging Hard voting = False': BaggingClf(hard_voting=False, weighted=False),
    'RandomSubspace Hard voting = True': RandomSubspaceEnsemble(hard_voting=True, base_estimator=DecisionTreeClassifier(), n_estimators=5),
    'RandomSubspace Hard voting = False': RandomSubspaceEnsemble(hard_voting=False, base_estimator=DecisionTreeClassifier(), n_estimators=5),

}
scores = np.zeros((len(clfs), n_splits * n_repeats))
for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clfs_name in enumerate(clfs):
        clf = clone(clfs[clfs_name])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        # print(accuracy_score(y[test], y_pred))
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))



alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("Statistically significantly better:\n", stat_better_table)
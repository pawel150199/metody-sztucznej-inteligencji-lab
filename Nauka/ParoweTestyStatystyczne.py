from statistics import mean
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel
from tabulate import tabulate


dataset = 'australian'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)


clfs = {
    'GNB': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=42),
}

n_split = 5
n_repeat = 2

scores = np.zeros((len(clfs), n_split*n_repeat))
rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

for fold_id, (train, test) in enumerate(rskf.split(X,y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

#zapis i odczyt z pliku
np.save('results', scores)
scores = np.load('results.npy')
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
print("t-statistic:\n\n", t_statistic_table, "\n\np-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Rónice statystyczne znaczące
significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)
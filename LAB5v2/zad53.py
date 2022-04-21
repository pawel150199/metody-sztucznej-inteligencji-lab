from statistics import mean
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel, ttest_ind
from tabulate import tabulate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from scipy.stats import rankdata
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import pandas as pd
from math import pi


clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

score = np.load('results.npy')
scores = score[1,:,:]
alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))



for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

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

#############################################################
"""wyniki testu parowego statystycznego na globalnych rangach"""
print("##############WYNIKI TESTU PAROWEGO WILIXONA################")


#Rangi
ranks = []
for ms in scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)



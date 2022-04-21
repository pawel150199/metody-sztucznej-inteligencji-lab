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
from sklearn.svm import SVC
from sklearn.base import clone
from scipy.stats import rankdata
from scipy.stats import ranksums
from scipy.stats import wilcoxon

scores = np.load('results.npy')

clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

mean = np.mean(scores, axis=1)
print(mean)

#Rangi
ranks = []
for ms in mean:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("")

#uśrednione Rangi
mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n\n", mean_ranks)

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = wilcoxon(mean[i], mean[j], zero_method="zsplit")

print(w_statistic)
headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\nw-statistic:\n\n", w_statistic_table, "\n\np-value:\n\n", p_value_table)






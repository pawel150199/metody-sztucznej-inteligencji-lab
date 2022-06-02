from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel

clf = DecisionTreeClassifier(random_state=2137)

preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=2137),
    'smote' : SMOTE(random_state=2137),
    'rus': RandomUnderSampler(random_state=2137),
}
metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

headers_1 = ["none", "ros", "smote","rus"]
n_column = np.array([["recall"], ["precision"], ["specificity"], ["f1"], ["g-mean"], ["bac"]])

headers = ["none", "ros", "smote","rus"]
names_column = np.array([["none"], ["ros"], ["smote"],['rus']])
#1
print("Dla pierwszego zbioru")
X, y = datasets.make_classification(
    n_samples=100,
    n_features=6,
    n_classes=2,
    weights=[0.17,0.83],
    random_state=2137,
)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)



scores = np.mean(scores, axis=1).T


table = np.concatenate((n_column, scores), axis=1)
table = tabulate(table, headers_1, floatfmt=".2f")
print("dla zbioru 1:\n", table)

alfa = .05
t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))


for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)

#2
print("Dla drugiego zbioru")
X, y= datasets.make_classification(
    n_samples=100,
    n_features=6,
    n_classes=2,
    weights=[0.1,0.9],
    random_state=2137,
    flip_y=.05
)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)


scores = np.mean(scores, axis=1).T


table = np.concatenate((n_column, scores), axis=1)
table = tabulate(table, headers_1, floatfmt=".2f")
print("dla zbioru 2:\n", table)

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))


for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)
#3
print("Dla trzeciego zbioru")

X, y = datasets.make_classification(
    n_samples=1000,
    n_features=6,
    n_classes=2,
    weights=[0.01,0.99],
    random_state=2137,
)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)


scores = np.mean(scores, axis=1).T


table = np.concatenate((n_column, scores), axis=1)
table = tabulate(table, headers_1, floatfmt=".2f")
print("dla zbioru 3:\n", table)

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))


for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)
#4
print("Dla czwartego zbioru")
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=6,
    n_clusters_per_class=1,
    n_classes=3,
    weights=[0.09,0.45,0.45],
    random_state=2137,
)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)


scores = np.mean(scores, axis=1).T


table = np.concatenate((n_column, scores), axis=1)
table = tabulate(table, headers_1, floatfmt=".2f")
print("dla zbioru 4:\n", table)

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))


for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)
#5
print("Dla piatego zbioru")
X, y = datasets.make_classification(
    n_samples=100,
    n_features=20,
    n_informative=2,
    n_classes=2,
    random_state=2137,
)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X[train], y[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X[train], y[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test])

        for metric_id, metric in enumerate(metrics):
            scores[preproc_id, fold_id, metric_id] = metrics[metric](
                y[test], y_pred)


scores = np.mean(scores, axis=1).T


table = np.concatenate((n_column, scores), axis=1)
table = tabulate(table, headers_1, floatfmt=".2f")
print("dla zbioru 5:\n", table)

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))


for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table)
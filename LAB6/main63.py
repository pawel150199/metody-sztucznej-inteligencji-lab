from zad63  import RandomSubspaceEnsemble
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from zad63 import RandomSubspaceEnsemble
from zad62 import BaggingClassifier2
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from scipy.stats import ttest_ind

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
scores = np.zeros((len(datasets), len(clfs), n_split*n_repeat))
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
            scores[data_id, clf_id, fold_id] = accuracy_score(y[test], y_pred)


alpha=.05
m_fmt="%.3f"
std_fmt=None
nc="---"
db_fmt="%s"
tablefmt="plain"
clfs = list(clfs.keys())
n_clfs = len(clfs)
mean_scores = np.mean(scores, axis=2)
stds = np.std(scores, axis=2)


t = []
for db_idx, db_name in enumerate(datasets):
    # Wiersz z wartoscia srednia
    t.append([db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :]])
    # Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
    if std_fmt:
        t.append([''] + [std_fmt % v for v in stds[db_idx, :]])
    # Obliczenie wartosci T i P
    T, p = np.array(
        [[ttest_ind(scores[db_idx, i, :],
            scores[db_idx, j, :])
        for i in range(len(clfs))]
        for j in range(len(clfs))]
    ).swapaxes(0, 2)
    _ = np.where((p < alpha) * (T > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                for i in range(n_clfs)]
    t.append([''] + [", ".join(["%i" % i for i in c])
                                     if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                                     for c in conclusions])
    
headers = ['metrics', 'datasets']
for i in clfs:
    headers.append(i)

print('\n\n',tabulate(t, headers))
from sklearn import datasets, naive_bayes
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from scipy.stats import ttest_ind
from sklearn.base import ClassifierMixin, BaseEstimator
from zad81 import SamplingClassifier

clfs = {
    'none': SamplingClassifier(base_estimator=GaussianNB()),
    'ros': SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=RandomOverSampler(random_state=123)),
    'smote' : SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=SMOTE(random_state=123)),
    'rus': SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=RandomUnderSampler(random_state=123)),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

datasets = ['data1','data2', 'data3', 'data4', 'data5']
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(datasets), len(clfs), n_splits * n_repeats, len(metrics)))

for data_id, data_name in enumerate(datasets):
        dataset = np.genfromtxt("datasets/%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        print(y)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
                for clf_id, clf_name in enumerate(clfs):
                    clf = clone(clfs[clf_name])
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    for m_id, m_name in enumerate(metrics):
                        mtr = metrics[m_name]
                        scores[data_id, clf_id, fold_id, m_id] = mtr(y[test],y_pred)
alpha=.05
m_fmt="%.3f"
std_fmt=None
nc="---"
db_fmt="%s"
tablefmt="plain"
metrics = list(metrics.keys())
clfs = list(clfs.keys())
n_clfs = len(clfs)
mean_scores = np.mean(scores, axis=2)
stds = np.std(scores, axis=2)
table = {}

for m_idx, m_name in enumerate(metrics):
    t = []
    for db_idx, db_name in enumerate(datasets):
        # Wiersz z wartoscia srednia
        t.append([db_fmt % db_name] + [m_fmt % v for v in mean_scores[db_idx, :, m_idx]])
        # Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
        if std_fmt:
            t.append([''] + [std_fmt % v for v in stds[db_idx, :, m_idx]])
        # Obliczenie wartosci T i P
        T, p = np.array(
            [[ttest_ind(scores[db_idx, i, :, m_idx],
                scores[db_idx, j, :, m_idx])
            for i in range(len(clfs))]
            for j in range(len(clfs))]
        ).swapaxes(0, 2)
        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                    for i in range(n_clfs)]

        t.append([''] + [", ".join(["%i" % i for i in c])
                                         if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                                         for c in conclusions])

        table.update({m_name: tabulate(
                        t, headers=['DATASET'] + clfs, tablefmt=tablefmt)})

        

    
for m_idx, m_name in enumerate(table):
    print('\n\nMetryka %s\n' % (m_name))
    print(table[m_name])



#headers = ['metrics', 'datasets']
#for i in clfs:
#    headers.append(i)
#print(headers)
#print(tabulate(t, headers))


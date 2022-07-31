import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from zad62 import BaggingClassifier2
from tabulate import tabulate
from sklearn.base import clone
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

datasets = ['banana', 'balance',]

#'appendicitis', 'iris', 'magic', 'sonar'

clfs = {
    'Bagging HV, W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=True, random_state=1234),
    'Bagging HV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=True, weight_mode=False, random_state=1234),
    'Bagging NHV W': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=True, random_state=1234),
    'Bagging NHV NW': BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=1410), hard_voting=False, weight_mode=False, random_state=1234),
}   
metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

n_repeat = 5
n_split = 2
scores = np.zeros((len(datasets), len(clfs), n_split * n_repeat, len(metrics)))
rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=1410)

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            for m_id, m_name in enumerate(metrics):
                metric = metrics[m_name]
                scores[data_id, clf_id, fold_id, m_id] = metric(y[test],y_pred)

alpha=.05
m_fmt="%.3f"
std_fmt=None
nc="---"
db_fmt="%s"
tablefmt="plain"
clfs = list(clfs.keys())
n_clfs = len(clfs)
mean_ranks = []
ranks = []
metrics = list(metrics.keys())
mean_scores=np.mean(scores, axis=2)
for m, metric in enumerate(metrics):
    scores_ = mean_scores[:,:,m]
    rank=[]
    for row in scores_:
        rank.append(rankdata(row).tolist())
    ranks.append(rank)

ranks = np.array(ranks)
mean_ranks=np.mean(ranks, axis=1)
print(mean_ranks)
length=len(clfs)
s = np.zeros((length, length))
p = np.zeros((length, length))
mean_scores=mean_scores[:,:,1]
for i in range(length):
    for j in range(length):
        s[i, j], p[i, j] = wilcoxon(mean_scores.T[i], mean_scores.T[j], zero_method="zsplit")
print(s)
print(p)


t = []
for m, metric in enumerate(metrics):
    metric_ranks = ranks[m,:,:]
    length = len(clfs)
    s = np.zeros((length, length))
    p = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            s[i, j], p[i, j] = wilcoxon(metric_ranks.T[i], metric_ranks.T[j], zero_method="zsplit")
    _ = np.where((p < alpha) * (s > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                   for i in range(length)]
    t.append(["%s" % metric] + ["%.3f" %
                                   v for v in
                                   mean_ranks[m]])
    # t.append([''] + [", ".join(["%i" % i for i in c])
    #                  if len(c) > 0 else nc
    #                  for c in conclusions])
    t.append([''] + [", ".join(["%i" % i for i in c])
                     if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                     for c in conclusions])

    
headers = ['metrics']
for i in clfs:
    headers.append(i)

print('\n\n',tabulate(t, headers))


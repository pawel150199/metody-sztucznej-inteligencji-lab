import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy import rand
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score

datasets = 'australian'

dataset = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clf = DecisionTreeClassifier(random_state=1410)
preprocs = {
    'None': None,
    'ROS': RandomOverSampler(random_state=1410),
    'SMOTE': SMOTE(random_state=1410),
    'RUS': RandomUnderSampler(random_state=1410),
    'CNN': CondensedNearestNeighbour(random_state=1410),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=1410)

scores = np.zeros((len(preprocs), n_repeats * n_splits , len(metrics)))

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

# Zapisanie wynikow
np.save('results', scores)

#kontynuacja
scores = np.load("results.npy")
scores = np.mean(scores, axis=1).T

metrics=["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']
methods=["None", 'ROS', 'SMOTE', 'RUS', 'CNN']
N = scores.shape[0]

# kat dla kazdej z osi
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# spider plot
ax = plt.subplot(111, polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)


# Dodajemy wlasciwe ploty dla kazdej z metod
for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# Dodajemy legende
plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# Zapisujemy wykres
plt.savefig("radar", dpi=200)

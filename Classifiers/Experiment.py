from RandomClassifier import RandomClassifier
from NClassifier import NClassifier
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from prettytable import PrettyTable as table


clfs = {
    "RC": RandomClassifier(),
    "kNN": NClassifier(),
}

data = [
    datasets.make_moons(),
    datasets.make_circles(),
    datasets.make_blobs(random_state=42),
]
n_data = len(data)
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234
)
scores = np.zeros((len(clfs), n_data, n_splits * n_repeats))

for i in range(0, len(data)):
    d = data[i]
    X = d[0]
    y = d[1]
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, i, fold_id] = accuracy_score(y[test], y_pred)

# zapisuje wartości srednie i odchylenie standardowe
mean_scores = np.mean(scores, axis=2).T
std_scores = np.std(scores, axis=2).T


# zapisanie wyników do tabel
data_names = ["moons", "circles", "blob"]
rc = []
knn = []
for i in range(0, n_data):
    rc.append(round(mean_scores[i, 0], 3))
    knn.append(round(mean_scores[i, 1], 3))

# zapisuwanie wartosci odchylenia  standardowego
s_scores = table()
s_scores.field_names = ("nazwa zbioru", "Random Classifier",  "kNN Clasifier")
rc_s = []
knn_s = []
for i in range(0, n_data):
    rc_s.append(round(std_scores[i, 0], 3))
    knn_s.append(round(std_scores[i, 1], 3))

for i in range(0, len(data_names)):
    s_scores.add_row([data_names[i], str(f"{rc[i]}\n({rc_s[i]})\n\n"), str(f"{knn[i]}\n({knn_s[i]})\n\n")])


print("\n\nWyniki doświadczenia: \n\n")
print(s_scores)

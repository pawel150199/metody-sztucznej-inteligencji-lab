from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2

#dane 
X, y = make_classification(n_samples = 500)

#klasyfikatory
clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=42),
}

#tablica z wartościami z rozkładu normalnego
rand = np.random.normal(size=X.shape[1])

#mnozenie elementów 
for i in range(X.shape[0]):
    for j in range(len(rand)):
        X[i, j] = X[i, j] * rand[j]


n_splits = 2
n_repeats = 5

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=42,
)

print(f"X shape: {X.shape}")#kształt oryginalny

"""
scores = np.zeros((len(clfs), n_splits*n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

print("\n############################\n")

"""
#selekcja danych
X = SelectKBest(k=4).fit_transform(X,y)

scoress = np.zeros((len(clfs), n_splits*n_repeats))
for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scoress[clf_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scoress, axis=1)
std = np.std(scoress, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

print(f"X shape po redukcji: {X.shape}")#kształt po redukcji
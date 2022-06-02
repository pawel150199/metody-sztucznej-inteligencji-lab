from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#dane 
X, y = make_classification(
    n_samples = 1000,
    n_classes = 10,
    n_informative = 10
)

#klasyfikatory
clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'SVC': SVC(),
}

#tablica z wartościami z rozkładu normalnego
rand = np.random.normal(size=X.shape[1])

#mnozenie elementów 
X = X*rand

n_splits = 2
n_repeats = 5

#walidacja krzyzowa
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=42,
)

#z normalizacją
scoress = np.zeros((len(clfs), n_splits*n_repeats))#macierz na wyniki
for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clfs[clf_name]
        scaler = StandardScaler()#inicjalizacja 
        scaler.fit(X[train])#skalowanie
        X_test = scaler.transform(X[test])
        X_train = scaler.transform(X[train])
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        scoress[clf_id, fold_id] = accuracy_score(y[test], y_pred)

mean = np.mean(scoress, axis=1)
std = np.std(scoress, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))
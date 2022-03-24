from zad21 import RandomClassifier
from zad22 import NClassifier
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

clfs = {
    'kNN': NClassifier(),
}
'''
data = [datasets.make_moons(), datasets.make_circles(), datasets.make_blobs()]

n_data = len(data)
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
scores = np.zeros((len(clfs), n_data, n_splits*n_repeats))

for i in range(0,len(data))
    y = data[i, 1].astype(int)
    print(X)
    print(y)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            print(y_pred, "\n")
            print(y[test])
            scores[clf_id, i, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)
'''
data = datasets.make_blobs()
X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 2
    )
#własny klasyfikator
clf = NClassifier()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)
print(f"Accuracy Score for own estimator: {score.round(2)}\n")
from zad21 import RandomClassifier
from zad22 import NClassifier
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from prettytable import PrettyTable as table


clfs = {
    'RC': RandomClassifier(),
    'kNN': NClassifier(),
}

data = [datasets.make_moons(random_state=42), datasets.make_circles(random_state=42), datasets.make_blobs(random_state=42)]
n_data = len(data)
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
scores = np.zeros((len(clfs), n_data, n_splits*n_repeats))

for i in range(0,len(data)):
    d = data[i]
    X = d[0]
    y = d[1]
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, i, fold_id] = accuracy_score(y[test], y_pred)

#zapisuje wartości srednie i odchylenie standardowe
mean_scores = np.mean(scores, axis=2).T
std_scores = np.std(scores, axis=2).T
print(mean_scores)


#zapisanie wyników do tabel
data_names = ['moons', 'circles', 'blob']
m_scores = table()
m_scores.field_names = ('nazwa zbioru', 'Random Classifier', 'k Neighbours Classifier' )
rc = []
knn = []
for i in range(0, n_data):
    rc.append(round(mean_scores[i,0],3))
    knn.append(round(mean_scores[i,1],3))   

for i in range(0, len(data_names)):
    m_scores.add_row([data_names[i], rc[i], knn[i]])

s_scores = table()

print("Mean Scores: \n\n")
print(m_scores)
    




"""
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

#Ten gorszy bo obcy klasyfikator
clf2 = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
clf2.fit(X_train, y_train)
predict2 = clf2.predict(X_test)
score2 = accuracy_score(y_test, predict2)

print(f"Accuracy Score for own estimator: {score.round(2)}\n")
print(f"Accuracy Score KN: {score2.round(2)}\n")
"""

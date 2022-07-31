import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#dane 
datasets = ['appendicitis', 'balance', 'banana', 'bupa', 'glass',
            'iris', 'led7digit', 'magic', 'phoneme', 'ring', 'segment',
            'sonar', 'spambase', 'texture', 'twonorm', 'wdbc',
            'winequality-red', 'winequality-white', 'yeast']


#klasyfikatory
clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}
n_splits = 2#ilość foldów
n_repeats = 5#powtórzenia
n_datasets = len(datasets)#ilość zbiorów danych
scores = np.zeros((n_datasets, n_splits*n_repeats, len(clfs)))#wyniki w 3 wymiarowej macierzy
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1410)


for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X,y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clfs[clf_name]
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[data_id, fold_id, clf_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)


import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#pobranie danych z wcześniej zapisanego pliku 
dataset = np.genfromtxt("zadanie11.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)


#podzielenie zbioru na zbiór testowy i uczący
X_train, X_test, y_train, y_test = train_test_split(

    X, y,
    test_size = 0.7,
    random_state=1234,
)


kf = KFold(n_splits=5, shuffle = True, random_state=1234)
scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

#Wyniki walidacji krzyzowej

mean_score = np.mean(scores)
std_score = np.std(scores)
print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))
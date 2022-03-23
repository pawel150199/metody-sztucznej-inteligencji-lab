import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

X, y = datasets.make_classification(
    n_classes = 2,
    n_features = 2,
    n_informative = 2,
    flip_y = 0.08,
    n_redundant = 0,
    n_repeated = 0,
    random_state = 1410,
    n_samples = 400

)
print(X.shape)
fig, ax = plt.subplots(1,1, figsize = (10,5))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
ax.set_xlabel('feature 0')
ax.set_ylabel('feature 1')
ax.set_title('Datasets')
plt.tight_layout()
plt.show()

data = np.concatenate((X, y[:, np.newaxis]), axis=1)
np.savetxt('zadanie.csv', data, delimiter=",")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1410
)
clf = GaussianNB()
clf.fit(X_train, y_train)
classes_probabilities=clf.predict_proba(X_test)
predict = np.argmax(classes_probabilities, axis=1)
score = accuracy_score(y_test, predict)
print("Accuracy score:", score)

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr")
ax[0].set_xlabel('feature 0')
ax[0].set_ylabel('feature 1')
ax[0].set_title('Real Label')

ax[1].scatter(X_test[:, 0], X_test[:, 1], c=predict, cmap="bwr")
ax[1].set_xlabel('feature 0')
ax[1].set_ylabel('feature 1')
ax[1].set_title('Predict Label')

plt.tight_layout()
plt.show()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1410)
scores = []

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)

print("Accuracy score: %.3f, (%.3f)" % (mean_score, std_score))







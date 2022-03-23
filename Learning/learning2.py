import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from  sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(
    n_samples = 400,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_repeated = 0,
    weights = None,
    n_classes = 2,
    flip_y = .08,

)

print(X.shape)

fig, ax = plt.subplots(1,1, figsize=(10, 5))
ax.scatter(X[:, 0], X[:, 1], c = y, cmap = "bwr")
ax.set_title("Dataset")
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
plt.tight_layout()
plt.show()

data = np.concatenate((X, y[:, np.newaxis]), axis = 1)
np.savetxt('xd.csv', data , delimiter=",")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state = 42
)

clf = GaussianNB()
clf.fit(X_train, y_train)
classes_predict = clf.predict_proba(X_test)
predict = np.argmax(classes_predict, axis = 1)
score = accuracy_score(y, predict)

print(f"Accuracy.score: {round(score)}")

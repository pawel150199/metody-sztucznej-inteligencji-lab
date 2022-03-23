import matplotlib.pyplot as plt
from sklearn import datasets

X,y = datasets.make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 1,
    n_repeated = 0,
    n_redundant = 0,
    flip_y = .05,
    random_state = 1410,
    n_clusters_per_class = 1
)

plt.figure(figsize=(5,2.5))
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.tight_layout()
plt.show()
plt.savefig('new.png')

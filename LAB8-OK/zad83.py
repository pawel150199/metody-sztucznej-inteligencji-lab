from sklearn import datasets
import numpy as np

# 1
X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.17,0.83],
    random_state=1234
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data1.csv', dataset, delimiter=',')

# 2
X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[1,99],
    random_state=1234
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data2.csv', dataset, delimiter=',')

# 3
X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[1, 9],
    flip_y=.05,
    random_state=1234
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data3.csv', dataset, delimiter=',')

# 4
X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.09, 0.45, 0.45],
    random_state=1234
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data4.csv', dataset, delimiter=',')

# 4
X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=3,
    random_state=1234
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data5.csv', dataset, delimiter=',')

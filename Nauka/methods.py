from sklearn.datasets import make_classification, make_circles
import numpy as np

q = 512
r = 5

def dataset():
    #return make_circles(n_samples=30, noise=.1, factor=.5)
    X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, random_state=90210, weights=(.5, .5))

    return X, y

def probers():
    lsa = np.linspace(-r,r,q)
    XX = np.reshape(np.array(np.meshgrid(lsa, lsa)), (2,-1)).T
    return XX

from os import SCHED_OTHER
from tabnanny import check
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import DistanceMetric
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from numpy.random import MT19937
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



class RandomClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, random_state=1410):
        """Inicjalizacja generatora"""
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Uczenie się"""
        #Check that X and y have correct shape
        X,y = check_X_y(X, y)
        #przechowujemy unikalne klasy problemu
        self.classes_ = np.unique(y)
        #przechowujemy X i y
        self.X_, self.y_ = X, y


        one = 0
        for i in range(0, len(X)):
            if y[i]>0:
                one += 1
        self.p_one = one/len(y) # proporcja między klasami
        #print(self.p_one)
        #print(1-self.p_one)

        
        return self

    def predict(self, X):
        """Dokonujemy predykcji"""
        check_is_fitted(self)#sprawdzam czy jest wywołana metoda fit
        X = check_array(X) 
        y_predict = []
        #zwracanie losowej etykiety
        for i in range(0, len(X)):
            ran_y = np.random.choice([min(self.y_), max(self.y_)], 1, p=[1-self.p_one, self.p_one])
            y_predict.append(ran_y)
        return np.array(y_predict)

X, y = make_classification(
    n_samples = 200,#liczba generowanych wzorców
    n_features = 2,#liczba atrybutów
    n_classes = 2,#liczba klas problemu 2->problem binarny
    n_repeated = 0,#brak powtórzeń
    n_redundant = 0,#brak cech zbędnych
    #flip_y = 0.08,#szum etykiet 8% ogółu wzorca
    #random_state=1500,
    #weights=None
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 2
)

score = []
for i in range(0, 1000):
    clf = RandomClassifier()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    score.append(accuracy_score(y_test, predict))

#print(f"Accuracy Score: {score.round(2)}")
x  = np.mean(score)
print(x)



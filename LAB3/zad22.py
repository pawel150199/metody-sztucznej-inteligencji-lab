import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KNeighborsClassifier



class NClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier closest neighbour """

    def __init__(self, demo_param='demo'):
        """Inicjalizacja generatora"""
        self.demo_param = demo_param

    def fit(self, X, y):
        """Uczenie się"""
        #Check that X and y have correct shape
        X,y = check_X_y(X, y)
        #przechowujemy unikalne klasy problemu
        self.classes_ = unique_labels(y)
        #przechowujemy X i y
        self.X_, self.y_ = X, y
        self.class_probe = []
        self.n = X.shape[1]
 
        return self

    def predict(self, X):
        """Dokonujemy predykcji"""
        check_is_fitted(self)#sprawdzam czy jest wywołana metoda fit
        X = check_array(X) 

        self.y_predict = []#przewisywane wartośc

        #zwracanie losowej etykiety
        for i in range(0, len(X)):
            """tutaj iteruje po długości zbioru testowego"""
            temp = []

            for j in range(0, len(self.y_)):
                """tutaj iteruje do  długości y test"""
                x = pow((self.X_[j,0]-X[i,0]), 2)
                y = pow((self.X_[j,1]-X[i,1]), 2)
                xd = np.sqrt(x+y) 
                temp.append(xd)
            mind = np.min(temp)
            ix = temp.index(mind)
            self.y_predict.append(self.y_[ix])
        return self.y_predict


if __name__=='__main__':
    X, y = make_classification(
        n_samples = 200,#liczba generowanych wzorców
        n_features = 2,#liczba atrybutów
        n_classes = 2,#liczba klas problemu 2->problem binarny
        n_repeated = 0,#brak powtórzeń
        n_redundant = 0,#brak cech zbędnych
        #flip_y = 0.08,#szum etykiet 8% ogółu wzorca
        random_state=1500,
        n_informative=2
        #weights=None
    )

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




from tabnanny import check
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.random import RandomState


class RandomClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, random_state=None):
        """Inicjalizacja generatora"""
        self.random_state = RandomState()

    def fit(self, X, y):
        """Uczenie się"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # przechowujemy unikalne klasy problemu
        self.classes_ = np.unique(y)
        # przechowujemy X i y
        self.X_, self.y_ = X, y
        # liczba klas problemu
        self.n_features = len(np.unique(y))
        return self

    def predict(self, X):
        """Dokonujemy predykcji"""
        check_is_fitted(self)  # sprawdzam czy jest wywołana metoda fit
        X = check_array(X)
        # zwracanie losowej etykiety
        y_predict = self.random_state.choice(self.n_features, len(X))
        return np.array(y_predict)


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=200,  # liczba generowanych wzorców
        n_features=2,  # liczba atrybutów
        n_classes=2,  # liczba klas problemu 2->problem binarny
        n_repeated=0,  # brak powtórzeń
        n_redundant=0,  # brak cech zbędnych
        # flip_y = 0.08,#szum etykiet 8% ogółu wzorca
        # random_state=1500,
        # weights=None
        n_informative=2,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    clf = RandomClassifier()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    score = accuracy_score(y_test, predict)

    print(f"Accuracy Score: {score.round(2)}")

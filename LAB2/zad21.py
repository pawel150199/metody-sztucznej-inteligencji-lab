import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

#generowanie danych 
X,y = datasets.make_classification(
    n_samples = 400,#liczba generowanych wzorców
    n_features = 2,#liczba atrybutów
    n_classes = 2,#liczba klas problemu 2->problem binarny
    n_repeated = 0,#brak powtórzeń
    n_redundant = 0,#brak cech zbędnych
    flip_y = 0.08,#szum etykiet 8% ogółu wzorca
    random_state=1500,
    weights=None
)

print(X.shape) # sprawdzam czy się zgadza

print(X)
print(y)

#generowanie scatterplota
"""
fig, ax = plt.subplots(1,1,figsize=(9,8))
ax.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_title("Dataset")
plt.tight_layout()
plt.show()


#dodanie do tablicy X nowej kolumny z etykietami
dataset = np.concatenate((X,y[:, np.newaxis]), axis=1)

#zapis danych do pliku csv
np.savetxt(
    "zad21.csv",
    dataset,
    delimiter=",",
    fmt = ['%.5f' for i in range(X.shape[1])] + ["%i"],
)
print(dataset)
"""
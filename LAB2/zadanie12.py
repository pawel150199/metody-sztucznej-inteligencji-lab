from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#pobranie danych z wcześniej zapisanego pliku 
dataset = np.genfromtxt("zadanie11.csv", delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

#podzielenie zbioru na zbiór testowy i uczący
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 1234
)
#inicjalizacja i budowa modelu klasyfikacji/ Naiwny klasyfikator bayesowski
clf = GaussianNB()
clf.fit(X_train, y_train)

#wyznacznie macierzy wsparcia
class_probabilities = clf.predict_proba(X_test)
#print(f"Macierz wsparć: \n{class_probabilities}\n")

#predykcja klasyfikatora dla zbioru testowego
predict = np.argmax(class_probabilities, axis=1)
print(f"Predykcja klasyfikatora: \n{predict}\n")

#porównanie wartości rzeczywistych z predykcyjnymi
print(f"Wartości rzeczywiste: \n{y_test}\n")
print(f"Wartości przewidziane: \n{predict}\n")

#ocena jakości modelu (occuracy)
score = accuracy_score(y_test, predict)

print(f"Occuracy: {score.round(2)}\n")

#generowanie wykresu
fig, ax = plt.subplots(1,2,figsize=(10,5))


#rzeczywiste etykiety
ax[0].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr')
ax[0].set_xlabel('feature 0')
ax[0].set_ylabel('feature 1')
ax[0].set_title('Real labels')

#etykiety dotyczące predykcji
ax[1].scatter(X_test[:,0], X_test[:,1], c=predict, cmap='bwr')
ax[1].set_xlabel('feature 0')
ax[1].set_ylabel('feature 1')
ax[1].set_title('Predict labels')

plt.tight_layout()
plt.show()
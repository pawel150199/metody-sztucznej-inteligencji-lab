import numpy as np
import pandas as pd
import sklearn as sk

A = np.ones((3, 5))
columns_sum = A.sum(axis=0)
row_sum = A.sum(axis=1)
B = np.arange(20)

print(f"tablica A o wymiarze 3x5:\n\n{A}\n")
print(f"Suma kolumn:\t{columns_sum}\n")
print(f"Suma wierszy:\t{row_sum}\n")
print(f"Tablica B: 20 elementowa, jednowymiarowa:\t{B}\n")
print(f"Wyświetlenie elementów 10-12 z tabeli B:\t{B[9:12]}\n")

import numpy as np
from strlearn import evaluators, ensembles, streams, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

stream = streams.StreamGenerator(n_chunks=200,
                                    chunk_size=500,
                                    n_classes=2,
                                    n_drifts=1,
                                    n_features=10,
                                    random_state=12345)


clfs = [
    ensembles.SEA(GaussianNB(), n_estimators=10),
    MLPClassifier(hidden_layer_sizes=(10)),
]
clf_names = [
    "SEA",
    "MLP"
]   

# Wybrana metryka
metrics = [metrics.f1_score,
           metrics.geometric_mean_score_1]

# Nazwy metryk
metrics_names = ["F1 score",
                 "G-mean"]
# Inicjalizacja ewaluatora
evaluator = evaluators.TestThenTrain(metrics)
# Uruchomienie
evaluator.process(stream, clfs)
# Rysowanie wykresu
fig, ax = plt.subplots(1, len(metrics), figsize=(12,4))
for m, metrics in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0,1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.show()

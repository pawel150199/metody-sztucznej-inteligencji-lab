import numpy as np
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from tabulate import tabulate

for i in range (0,5):
    stream1_1 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           n_features=10,
                                           random_state=11255)

    stream1_2 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           n_features=10,
                                           random_state=12134)

    stream1_3 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           n_features=10,
                                           random_state=32154)

    stream1_4 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           n_features=10,
                                           random_state=52223)

    stream1_5 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           n_features=10,
                                           random_state=54321)


    stream2_1 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           n_features=10,
                                           random_state=11255)

    stream2_2 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           n_features=10,
                                           random_state=12134)

    stream2_3 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           n_features=10,
                                           random_state=32154)

    stream2_4 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           n_features=10,
                                           random_state=52223)

    stream2_5 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           n_features=10,
                                           random_state=54321)


    stream3_1 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           incremental=True,
                                           n_features=10,
                                           random_state=11255)

    stream3_2 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           incremental=True,
                                           n_features=10,
                                           random_state=12134)

    stream3_3 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           incremental=True,
                                           n_features=10,
                                           random_state=32154)

    stream3_4 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           incremental=True,
                                           n_features=10,
                                           random_state=52223)

    stream3_5 = sl.streams.StreamGenerator(n_chunks=500,
                                           chunk_size=250,
                                           n_classes=2,
                                           n_drifts=5,
                                           concept_sigmoid_spacing=5,
                                           incremental=True,
                                           n_features=10,
                                           random_state=54321)



streams1 = [stream1_1, stream1_2, stream1_3, stream1_4, stream1_5]
streams2 = [stream2_1, stream2_2, stream2_3, stream2_4, stream2_5]
streams3 = [stream3_1, stream3_2, stream3_3, stream3_4, stream3_5]

drift_types = np.array(["Sudden", "Gradual", "Incremental"])
replications = [["Replication 1"], ["Replication 2"], ["Replication 3"], ["Replication 4"], ["Replication 5"]]


scores1 = []
scores2 = []
scores3 = []


clfs = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
]

clf_names = [
    "SEA",
]

# Wybrana metryka
metrics = [sl.metrics.f1_score]

# Nazwy metryk
metrics_names = ["F1 score"]


# Inicjalizacja ewaluatora
evaluator = sl.evaluators.TestThenTrain(metrics)


# Uruchomienie
for i in range(len(streams1)):
    evaluator.process(streams1[i], clfs)
    scores1.append(evaluator.scores)
    evaluator.process(streams2[i], clfs)
    scores2.append(evaluator.scores)
    evaluator.process(streams3[i], clfs)
    scores3.append(evaluator.scores)



fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax.set_title(metrics_names)
    ax.set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax.plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax.legend()
plt.show()


fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax.set_title(metrics_names)
    ax.set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax.plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax.legend()
plt.show()


fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax.set_title(metrics_names)
    ax.set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax.plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax.legend()
plt.show()

means = []

for i in range(len(streams1)):
    means.append(np.mean(scores1[i]))

for i in range(len(streams2)):
    means.append(np.mean(scores2[i]))

for i in range(len(streams3)):
    means.append(np.mean(scores3[i]))


table1 = [[]]

for i in range(len(drift_types)):
    i=i*5
    temp_tab = []
    for j in range(len(streams1)):
        temp_tab.append(means[i+j])
    table1.append(temp_tab)


final_table = np.concatenate((drift_types, table1), axis=0)
final_table = tabulate(table1, replications, floatfmt=".3f")

print("Tabela srednich:\n", final_table)
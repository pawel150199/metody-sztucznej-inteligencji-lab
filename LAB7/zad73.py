import numpy as np
import strlearn as sl
import matplotlib.pyplot as plt
from strlearn.ensembles import UOB, SEA
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ttest_rel
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import rankdata
from scipy.stats import ttest_ind
from tabulate import tabulate
from strlearn.metrics import geometric_mean_score_1


#granularny
stream3_1 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=51555)

stream3_2 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=42231)

stream3_3 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=41255)

stream3_4 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=22231)

stream3_5 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=45122)

stream3_6 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=31244)

stream3_7 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=55555)

stream3_8 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=51235)

stream3_9 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=44123)


stream3_10 = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=150,
                                    n_classes=2,
                                    n_drifts=6,
                                    concept_sigmoid_spacing=5,
                                    n_features=10,
                                    random_state=12321)


streams = [stream3_1, stream3_2, stream3_3, stream3_4, stream3_5,stream3_6, stream3_7, stream3_8, stream3_9, stream3_10]


clfs = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
    sl.ensembles.OOB(GaussianNB(), n_estimators=10),
]

clfs_names = [
    "SEA",
    "OOB",
]

preprocs = {
    'ros': RandomOverSampler(random_state=4413),
    'rus': RandomUnderSampler(random_state=4413),
}

# Wybrana metryka
metric = [sl.metrics.f1_score]

# Nazwy metryk
metric_name = ["F1 score"]


# Inicjalizacja ewaluatora
evaluator = sl.evaluators.TestThenTrain(metric)

scores = []

# Uruchomienie
for i in range(len(streams)):
    evaluator.process(streams[i], clfs)
    scores.append(evaluator.scores)

means = []

means = np.mean(scores, axis=0)
means = np.mean(means, axis=2)

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))
names_column = np.expand_dims(np.array(clfs_names), axis=1)


print("Testy parowe dla zbioru 1:")

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(means[i], means[j])
#print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, clfs_names, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, clfs_names, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), clfs_names)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), clfs_names)
print("Statistical significance (alpha = 0.05):\n", significance_table)
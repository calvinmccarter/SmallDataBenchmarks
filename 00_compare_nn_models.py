"""
Which baseline models are best?
"""

import math
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import load_data

from simple_transformers import TwoLayerNetClassifier


N_JOBS = 1 #4 * 4 * 9


database = pd.read_json("database.json").T
# note: this meta-dataset has swallowed information about what's categorical and what isn't
# which means we are just going to be using each feature as continuous even though it
# may not be
database = database[database.nrow >= 50]


def evaluate_pipeline_helper(X, y, pipeline, param_grid, scoring="roc_auc_ovr_weighted", random_state=0):
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring=scoring, n_jobs=N_JOBS)
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring=scoring, n_jobs=N_JOBS)
    return nested_score


def define_and_evaluate_pipelines(X, y, random_state=0):
    N_s, N_i = X.shape
    N_o = len(np.unique(y))
    strats = [
        math.sqrt(N_i * N_o),
        N_s / (1. * (N_i + N_o)),
        N_s / (2. * (N_i + N_o)),
        N_s / (5. * (N_i + N_o)),
        N_s / (10. * (N_i + N_o)),
        N_i,
        N_o,
        (2/3)*N_i + N_o,
    ]
    n_hiddens = np.geomspace(min(strats), max(strats), 7).astype('int')

    # Fully-connected
    pipeline1 = TwoLayerNetClassifier(method="fc-relu")
    param_grid1 = {
        "n_hidden": n_hiddens,
    }

    # Scaled dot-product cross-attention
    pipeline2 = TwoLayerNetClassifier(method="dot")
    param_grid2 = {
        "n_hidden": n_hiddens,
    }

    pipeline3 = TwoLayerNetClassifier(method="idw")
    param_grid3 = {
        "n_hidden": n_hiddens,
    }

    """ this did the best
    # K nearest neighbors
    pipeline3 = KNeighborsClassifier(weights="distance")
    param_grid3 = {
        "n_neighbors": [1, 2, 3, 5, 8, 13, 21],
    }
    """

    nested_scores1 = evaluate_pipeline_helper(X, y, pipeline1, param_grid1, random_state=random_state)
    nested_scores2 = evaluate_pipeline_helper(X, y, pipeline2, param_grid2, random_state=random_state)
    nested_scores3 = evaluate_pipeline_helper(X, y, pipeline3, param_grid3, random_state=random_state)

    return nested_scores1, nested_scores2, nested_scores3


# run models on all datasets
results1 = []
results2 = []
results3 = []
evaluated_datasets = []
times = []
for i, dataset_name in enumerate(database.index.values):
    if dataset_name not in evaluated_datasets:
        X, y = load_data(dataset_name)
        # datasets might have too few samples per class
        if len(y) > 0 and np.sum(pd.value_counts(y) <= 15) == 0:
            np.random.seed(0)
            if len(y) > 10000:
                # subset to 10000 if too large
                random_idx = np.random.choice(len(y), 10000, replace=False)
                X = X[random_idx, :]
                y = y[random_idx]
            y=y.astype('int')
            print("starting:", dataset_name, X.shape)
            start = time.time()
            nested_scores1, nested_scores2, nested_scores3 = define_and_evaluate_pipelines(X, y)
            results1.append(nested_scores1)
            results2.append(nested_scores2)
            results3.append(nested_scores3)
            elapsed = time.time() - start
            evaluated_datasets.append(dataset_name)
            times.append(elapsed)
            print("done. elapsed:", elapsed)
            print("scores:", np.mean(nested_scores1), np.mean(nested_scores2), np.mean(nested_scores3))

#
results1 = np.array(results1)
results2 = np.array(results2)
results3 = np.array(results3)
evaluated_datasets = np.array(evaluated_datasets)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open("results/00_compare_baseline_models.pickle", "wb") as f:
    pickle.dump((results1, results2, results3, evaluated_datasets, times), f)

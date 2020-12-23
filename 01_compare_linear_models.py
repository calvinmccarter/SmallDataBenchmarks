"""
Which linear models are optimal?
"""

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV


N_JOBS = 24


database = pd.read_json("database.json").T
# note: for now we will ignore those with missing values
# there are very few of them
# note: this meta-dataset has swallowed information about what's categorical and what isn't
# which means we are just going to be using each feature as continuous even though it
# may not be
database = database[database.mv == 0]


def load_data(data_name):
    data, meta = arff.loadarff(f"datasets/{data_name}.arff")
    df = pd.DataFrame(data).apply(lambda x: pd.to_numeric(x, errors="ignore"))
    X = pd.get_dummies(df.loc[:, df.columns != "Class"]).values
    unique_labels = df["Class"].unique()
    labels_dict = dict(zip(unique_labels, range(len(unique_labels))))
    df.loc[:, "Class"] = df.applymap(lambda s: labels_dict.get(s) if s in labels_dict else s)
    y = df["Class"].values
    return X, y


def evaluate_pipeline_helper(X, y, pipeline, p_grid, random_state=0):
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    clf = GridSearchCV(
        estimator=pipeline, param_grid=p_grid, cv=inner_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS
    )
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS)
    return nested_score


def define_and_evaluate_pipelines(X, y, random_state=0):
    pipeline1 = Pipeline(
        [("scaler", MinMaxScaler()), ("svc", SVC(kernel="linear", probability=True, random_state=random_state))]
    )
    param_grid1 = {
        "svc__C": np.logspace(-7, 2, 10),
    }

    pipeline2 = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("logistic", LogisticRegression(solver="saga", max_iter=10000, random_state=random_state)),
        ]
    )
    param_grid2 = {
        "logistic__C": np.logspace(-7, 2, 10),
    }

    pipeline3 = BaggingClassifier(
        Pipeline([("scaler", MinMaxScaler()), ("ridge", RidgeClassifier(solver="saga", random_state=random_state)),])
    )
    param_grid3 = {
        "base_estimator__ridge__alpha": np.logspace(-7, 2, 10),
    }

    nested_scores1 = evaluate_pipeline_helper(X, y, pipeline1, param_grid1, random_state=random_state)
    nested_scores2 = evaluate_pipeline_helper(X, y, pipeline2, param_grid2, random_state=random_state)
    nested_scores3 = evaluate_pipeline_helper(X, y, pipeline3, param_grid3, random_state=random_state)

    return nested_scores1, nested_scores2, nested_scores3


# run models on all datasets
results1 = []
results2 = []
results3 = []
evaluated_datasets = []
for i, dataset_name in enumerate(database.index.values):
    X, y = load_data(dataset_name)
    if len(y) > 25 and len(y) < 1000:
        print(i, dataset_name, len(y))
        nested_scores1, nested_scores2, nested_scores3 = define_and_evaluate_pipelines(X, y)
        results1.append(nested_scores1)
        results2.append(nested_scores2)
        results3.append(nested_scores3)
        evaluated_datasets.append(dataset_name)

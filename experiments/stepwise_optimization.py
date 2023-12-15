from functools import partial
import os
import numpy as np
import pickle as pkl
from typing import Any, Callable, Sequence

from optuna import Trial, create_study
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import estimator_html_repr

from src.optuna_helper_functions import (
    get_lgbm_classifier,
    get_top_n_percent_trials,
    suggest_lgbm_step_one,
    suggest_lgbm_step_two,
    suggest_lgbm_step_three,
    suggest_lgbm_step_four,
    StepwiseStudy,
)
from src.utils import (
    ScikitLearnClassifier,
    get_numerical_columns,
    get_categorical_columns,
    get_function_max,
    get_model_performance,
)

DATA_FOLDER = os.path.join("data", "raw_data")
RESULTS_FOLDER = os.path.join("results", "stepwise_optimization")


def step_one_objective(
    trial: Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    aggregation: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)
    model = get_lgbm_classifier(
        trial,
        numerical_columns,
        categorical_columns,
        suggest_function=suggest_lgbm_step_one,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=make_scorer(brier_score_loss, needs_proba=True),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )

    return aggregation(scores)


def step_two_objective(
    trial: Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    aggregation: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)
    model = get_lgbm_classifier(
        trial,
        numerical_columns,
        categorical_columns,
        suggest_function=suggest_lgbm_step_two,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=make_scorer(brier_score_loss, needs_proba=True),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )

    return aggregation(scores)


def step_three_objective(
    trial: Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    aggregation: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)
    model = get_lgbm_classifier(
        trial,
        numerical_columns,
        categorical_columns,
        suggest_function=suggest_lgbm_step_three,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=make_scorer(brier_score_loss, needs_proba=True),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )

    return aggregation(scores)


def step_four_objective(
    trial: Trial,
    X: pd.DataFrame,
    y: np.ndarray,
    aggregation: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)
    model = get_lgbm_classifier(
        trial,
        numerical_columns,
        categorical_columns,
        suggest_function=suggest_lgbm_step_four,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=make_scorer(brier_score_loss, needs_proba=True),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )

    return aggregation(scores)


def instantiate_model(
    trial: Trial,
    params: dict[str, Any],
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ScikitLearnClassifier:
    suggest_params = partial(suggest_lgbm_step_four, previous_params=params)
    model = get_lgbm_classifier(
        trial,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        suggest_function=suggest_params,
    )

    return model


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(DATA_FOLDER, "data.parquet"))
    X = df.drop("target", axis=1)
    y = df.target.values.ravel()

    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=False)

    objectives = map(
        lambda objective: partial(objective, X=X_train, y=y_train, aggregation=np.max),
        [
            step_one_objective,
            step_two_objective,
            step_three_objective,
            step_four_objective,
        ],
    )

    sampler = TPESampler(n_startup_trials=10, seed=42)

    study = StepwiseStudy(
        storage="sqlite:///results/optimization.db",
        study_name="stepwise_optimization",
        sampler=sampler,
        direction="maximize",
    )

    study.optimize(objectives, n_trials=10)

    best_10_percent_trials = get_top_n_percent_trials(study, percentage=10)
    best_10_percent_models = [
        *map(
            lambda trial: instantiate_model(trial).fit(X_train, y_train),
            best_10_percent_trials,
        )
    ]

    get_performance = partial(
        get_model_performance, X=X_test, y=y_test, function=brier_score_loss
    )
    best_model = get_function_max(best_10_percent_models, get_performance)

    with open(os.path.join(RESULTS_FOLDER, "optimized.pkl"), "wb") as f:
        pkl.dump(best_model, f)

    with open(os.path.join(RESULTS_FOLDER, "optimized.html"), "w") as f:
        f.write(estimator_html_repr(best_model))

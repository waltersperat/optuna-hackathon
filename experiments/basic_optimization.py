import os
from typing import Callable
import numpy as np

from optuna import Trial, create_study
import pandas as pd
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.optuna_helper_functions import get_model
from src.utils import get_numerical_columns, get_categorical_columns

DATA_FOLDER = os.path.join('data', 'raw_data')

def objective(trial : Trial, X : pd.DataFrame, y : np.ndarray, aggregation : Callable[[np.ndarray, np.ndarray], float]) -> float:
    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)
    model = get_model(trial, numerical_columns, categorical_columns)

    scores = cross_val_score(
        model, X, y, scoring=make_scorer(brier_score_loss, needs_proba=True)
    )

    return aggregation(scores)

if __name__=='__main__':
    df = pd.read_parquet(os.path.join(DATA_FOLDER, 'data.parquet'))
    
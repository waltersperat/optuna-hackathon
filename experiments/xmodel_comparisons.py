from functools import partial
import os
import pickle as pkl

from mlxtend.evaluate import combined_ftest_5x2cv
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from src.optuna_helper_functions import get_lgbm_classifier

from src.utils import get_categorical_columns, get_numerical_columns

DATA_FOLDER = os.path.join("data", "raw_data")

with open('results/basic_optimization/optimized.pkl', 'rb') as f:
    basic = pkl.load(f)

with open('results/stepwise_optimization/optimized.pkl', 'rb') as f:
    stepwise = pkl.load(f)

if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(DATA_FOLDER, "data.parquet"))
    X = df.drop("target", axis=1)
    y = df.target.values.ravel()

    numerical_columns = get_numerical_columns(X)
    categorical_columns = get_categorical_columns(X)

    instantiate_model = partial(
        get_lgbm_classifier,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=False)
    
    print(combined_ftest_5x2cv(basic, stepwise, X_train, y_train, scoring=brier_score_loss))
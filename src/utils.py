from typing import Any, Callable, Protocol, Sequence, TypeVar
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.utils.metaestimators import _safe_split

T = TypeVar("T")


class ScikitLearnClassifier(Protocol):
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """
    This function takes a dataframe and returns the numerical columns as a list.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        list[str]: The names of the numerical columns of the dataframe.
    """
    columns = df.select_dtypes(include="number").columns
    return [*columns]


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    This function takes a dataframe and returns the categorical columns as a list.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        list[str]: The names of the categorical columns of the dataframe.
    """
    numerical_columns = get_numerical_columns(df)
    return [*filter(lambda column: column not in numerical_columns, df.columns)]


def get_function_max(data: Sequence[T], function: Callable[[T], float]) -> T:
    """
    This function takes a sequence and a function, and returns the value of the
    sequence for which the function evaluation is greater.

    Args:
        data (Sequence[T]): An iterable with the inputs of the function.
        function (Callable[[T], float]): The function to use when evaluating.

    Returns:
        T: The element of the iterable for which the function is greater.
    """
    transformed_data = [*map(function, data)]
    best_index = np.argmax(transformed_data)
    return data[best_index]


def get_model_performance(
    model: ScikitLearnClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    This function takes a model, data and a metric evaluation function and
    uses the function to evaluate the model's performance on the given metric.

    Args:
        model (ScikitLearnClassifier): The model to use.
        X (pd.DataFrame): The data to use for predictions.
        y (np.ndarray): The ground truth.
        function (Callable[[np.ndarray, np.ndarray], float]): The scoring function.

    Returns:
        float: The result of evaluating the function on the model predictions.
    """
    return function(y, model.predict_proba(X)[:, 1])

def fit_predict(estimator : ScikitLearnClassifier, X_train, y_train, X_test, predict_proba : bool=True) -> np.ndarray:
    model = estimator.fit(X_train, y_train)
    if predict_proba:
        predictions = model.predict(X_test)
    else:
        predictions = model.predict_proba(X_test)
    return predictions

def cross_val_predict(estimator : ScikitLearnClassifier, X : pd.DataFrame, y : np.ndarray, cv : int | BaseCrossValidator, random_state : int=42) -> float:
    if isinstance(cv, int):
        cv_ = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_ = cv_
    
    datasets = map(lambda train, test: (_safe_split(estimator, X, y, train), _safe_split(estimator, X, y, test)), cv_.split(X, y))

    train_test_performances = map(
        lambda data: fit_predict(clone(estimator), data[0][0], data[0][1], data[1][0]),
        datasets
    )


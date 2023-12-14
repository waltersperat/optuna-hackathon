from typing import Any, Callable, Protocol, TypeVar
import numpy as np
import pandas as pd

T = TypeVar('T')

class ScikitLearnClassifier(Protocol):
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def predict_proba(self, X):
        ...

def get_numerical_columns(df : pd.DataFrame) -> list[str]:
    columns = df.select_dtypes(include='number').columns
    return [*columns]

def get_categorical_columns(df : pd.DataFrame) -> list[str]:
    numerical_columns = get_numerical_columns(df)
    return [*filter(lambda column: column not in numerical_columns, df.columns)]

def argmax(data : list[T], function : Callable[[T], float]) -> T:
    transformed_data = [*map(function, data)]
    best_index = np.argmax(transformed_data)
    return data[best_index]

def get_model_performance(model : ScikitLearnClassifier, X : pd.DataFrame, y : np.ndarray, function : Callable[[np.ndarray, np.ndarray], float]) -> float:
    return function(y, model.predict_proba(X)[:, 1])
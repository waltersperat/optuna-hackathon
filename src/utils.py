import pandas as pd

def get_numerical_columns(df : pd.DataFrame) -> list[str]:
    columns = df.select_dtypes(include='number').columns
    return [*columns]

def get_categorical_columns(df : pd.DataFrame) -> list[str]:
    numerical_columns = get_numerical_columns(df)
    return [*filter(lambda column: column not in numerical_columns, df.columns)]
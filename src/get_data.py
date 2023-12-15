import polars as pl

import numpy as np


def generate_classification_data(
    n_samples: int, n_numerical: int, n_categorical: int, n_categories=3
) -> pl.DataFrame:
    numerical_data = {
        f"num_{i}": np.random.randn(n_samples) for i in range(n_numerical)
    }
    categorical_data = {
        f"cat_{i}": np.random.choice(
            [f"Category_{j}" for j in range(n_categories)], n_samples
        )
        for i in range(n_categorical)
    }
    data = numerical_data | categorical_data
    target = pl.Series(values=np.random.choice([0, 1], n_samples), name="target")
    df = pl.DataFrame(data).hstack([target])

    return df


if __name__ == "__main__":
    df = generate_classification_data(
        n_samples=50_000, n_numerical=100, n_categorical=10, n_categories=10
    )
    df.write_parquet("data/raw_data/data.parquet")

import os

from optuna import create_study
import polars as pl



if __name__=='__main__':
    df = pl.read_parquet('data/raw_data/heroes.parquet')
    pass
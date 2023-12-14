import polars as pl
from sklego.datasets import load_heroes

if __name__=='__main__':
    df = pl.from_pandas(load_heroes(return_X_y=True, as_frame=True))
    df.write_parquet('data/raw_data/heroes.parquet')
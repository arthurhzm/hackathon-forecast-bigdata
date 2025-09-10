import pandas as pd
import os
import glob

def import_parquet_files(quantity=None):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
    data_path = os.path.join(BASE_DIR, "data")
    files = glob.glob(os.path.join(data_path, "*.parquet"))

    if quantity:
        files = files[:quantity]
        dfs = [pd.read_parquet(arq) for arq in files]
        df_final = pd.concat(dfs, ignore_index=True)
        return df_final
    else:
        return pd.read_parquet(files[0])

df = import_parquet_files()
print(df.head())
import pandas as pd
import numpy as np

df = pd.read_csv('drug_ml_info_binary.csv')
df = df.replace(-100, np.nan)
df = df.dropna(axis=1, how='all')
df = df.loc[:, (df != 0).any(axis=0)]

df.to_csv('drug_ml_info_binary_1.csv', index=False)
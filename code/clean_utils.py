"""
Last edited: 05-03-24
Author: Albert Cao
Description: functions to clean and process the data for subsequent model training
"""

import pandas as pd
import numpy as np

def clean_1():
    # Cleans 'drug_ml_info_binary.csv' by removing the -100s and 0s
    df = pd.read_csv('drug_ml_info_binary.csv')
    df = df.replace(-100, np.nan)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, (df != 0).any(axis=0)]

    df.to_csv('drug_ml_info_binary_1.csv', index=False)

def filter_ae(ae: str) -> pd.DataFrame:
    # Filters the dataset to focus on the specified ae
    # Drops columns which are 0 for the target
    # Turns the NaN into zeroes
    df = pd.read_csv('drug_ml_info_binary_1.csv')
    aes = ['diarrhoea', 'dizziness', 'headache', 'nausea', 'vomiting']
    aes.remove(ae)
    df = df.drop(aes, axis=1)
    df = df.drop(df[df[ae] == 0].index)
    df[ae] = df[ae].replace(np.nan, 0)
    return df
    

def normalize(df, ae: str) -> pd.DataFrame:
    # Normalizes the dataframe which has target as ae
    for col in df.columns:
        if col in ['drug', 'SMILES', ae]:
            continue
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def generate_datasets():
    aes = ['diarrhoea', 'dizziness', 'headache', 'nausea', 'vomiting']
    for ae in aes:
        df: pd.DataFrame = normalize(filter_ae(ae), ae)
        df.to_csv(f"data_normalized_{ae}.csv", index=False)

generate_datasets()


"""
Last edited: 03-29-24
Author: Albert Cao
Description: Convert drug_ml_info.csv to drug_ml_info_binary.csv to perform binary classification 
"""
import pandas as pd
df = pd.read_csv('drug_ml_info.csv')
labels = ['diarrhoea', 'nausea', 'vomiting', 'headache', 'dizziness']

def t(x):
    if (x == -100):
        return -100
    elif (x <= 3):
        return 0
    else:
        return 1

for label in labels:
    df[label] = df[label].apply(t)
    
df.to_csv('drug_ml_info_binary.csv', index=False)



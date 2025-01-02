"""
Last edited: 04-27-24
Author: Albert Cao
Description: Generate PCA and tSNE figures
"""

from PIL import Image
import os
import pandas as pd
import numpy as np
from code.utils import plot_dim_reduced, process

# GENERATE IMAGES

df_all = pd.read_csv('drug_ml_info_binary_1.csv')

names = ['diarrhoea', 'headache', 'nausea', 'vomiting', 'dizziness']
# data_list = [train_df, test_df, val_df, df_all]
# desc_list = ['train set', 'test set', 'validation set', 'full set']
data_list = [df_all]
desc_list = ['dataset']
header = ['bit' + str(i) for i in range(167)]

for (data, desc) in zip(data_list, desc_list):
    data = process(data)
    features = data[header]
    for name in names:
        labels = data[name].replace(np.nan, 'None')
        # assert features.shape[0] == len(labels)
        for dim_reduct in ['PCA', 't-SNE']:
            title = f'{dim_reduct} on {desc} of {name}'
            # if dim_reduct == 'PCA':
            plot_dim_reduced(features, labels, False, dim_reduct, title)


# aes = ['diarrhoea', 'dizziness', 'nausea', 'headache', 'vomiting']
aes = [['diarrhoea', 'dizziness'],
       ['nausea', 'headache'],
       ['vomiting']]
# sets = [['full', 'train'],
        # ['validation', 'test']]
pca_types = [True, False]
dims = [640, 480]

def getname(ae: str, set: str, pca_type: bool):
    if (pca_type):
        return f"./PCA/PCA on {set} of {ae}.png"
    else:
        return f"./t-SNE/t-SNE on {set} of {ae}.png"

def gettarget(ae: str, pca_type: bool):
    if (pca_type):
        return f"./PCA/PCA_{ae}.png"
    else:
        return f"./t-SNE/t-SNE_{ae}.png"

for pca_type in pca_types:
    img = Image.new('RGB', (dims[0]*2, dims[1]*3), (255, 255, 255))
    for i in range(len(aes)):
        for j in range(len(aes[i])):
            im = Image.open(getname(aes[i][j], 'dataset', pca_type))
            img.paste(im, (dims[0]*j, dims[1]*i))
    img.save(gettarget("new", pca_type))

from utils import plot_morgan, get_morgan, pairwise_similarity
import pandas as pd

df = pd.read_csv(f'drug_ml_info_binary.csv')
# plot_morgan(df, 'Tanimoto on entire dataset')

sim_list = pairwise_similarity(get_morgan(df))
name_list = df['drug'].tolist()


num = len(name_list)
assert num == len(sim_list)

matrix = []

for i in range(num+1):
    matrix.append([])

matrix[0].append(0)
matrix[0] += name_list

for i in range(1, num+1):
    matrix[i].append(name_list[i-1])
    for j in range(num):
        matrix[i].append(sim_list[i-1][j])

import csv

with open('morgan.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(matrix)

drug1 = 'ruxolitinib'
drug2 = 'tofacitinib'

print(f'{matrix[129][592]}')
for i in range(num+1):
    if drug1 == matrix[i][0]:
        for j in range(num+1):
            if drug2 == matrix[0][j]:
                print(f'{matrix[i][j]}')
                break 
        break

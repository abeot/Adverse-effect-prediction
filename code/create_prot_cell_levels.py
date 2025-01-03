"""
Last edited: 09-16-23
Author: Albert Cao
Description: Create cleaned data for protein expression levels in cells
"""

# File to do stuff for proteins
import pandas as pd
import json
from tqdm import tqdm

# Get prot cell levels
df = pd.read_table('normal_tissue.tsv')
print(df.columns)
print(df[df.isna().any(axis=1)])

with open("hpa_gene_seqs.json") as f:
    prot_dict = json.load(f)
    prots = set(prot_dict.keys())
print(len(prots))

df['Tissue cell type'] = df['Tissue'] + " " + df['Cell type']

cells = df['Tissue cell type'].unique().tolist()
correct_cells = set()
for cell in cells:
    cell = str(cell).lower()
    if 'gradient' not in cell:
        correct_cells.add(cell)
cells = correct_cells
print(f"Number of unique tissue cell types: {len(cells)}")

# Export the data

prot_levels = {}
for prot in prots:
    prot_levels[prot] = {}

for prot in prots:
    for cell in cells:
        prot_levels[prot][cell] = -100

def convert(level: str):
    if level == 'Low':
        return 1
    elif level == 'Medium':
        return 2
    elif level == 'High':
        return 3
    elif level == 'Not detected':
        return 0
    else:
        return -100

for ind in tqdm(df.index, total=df.shape[0],
                desc='Get protein expression info'):
    prot = df['Gene name'][ind]
    if prot not in prots:
        continue
    cell = df['Tissue cell type'][ind]
    if cell not in cells:
        continue
    level = df['Level'][ind]

    prot_levels[prot][cell] = convert(level)
print(f"Protein size: {len(prot_levels.keys())}")
print(f"Cell type size: {len(prot_levels['TSPAN6'].keys())}")
with open("prot_cell_levels.json", "w") as f:
    json.dump(prot_levels, f, indent=4)

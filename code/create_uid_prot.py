"""
Last edited: 09-15-23
Author: Albert Cao
Description: Create dictionaries to map proteins from different datasets
"""

import pandas as pd

hpa = pd.read_csv('proteinatlas.tsv', sep='\t')
uniprot = pd.read_csv('uniprotkb_reviewed_true_AND_model_organ_2023_08_23.tsv', sep='\t')

uid_map = dict()
for ind in uniprot.index:
    uid = uniprot['Entry'][ind]
    name = uniprot['Protein names'][ind]
    uid_map[uid] = name

map = dict()
uid_to_prot = dict()
for ind in hpa.index:
    gene = hpa['Gene'][ind]
    uid = hpa['Uniprot'][ind]
    if pd.isna(uid):
        print(f"Could not find UID for gene {gene}")
        map[gene] = ""
        continue
    uid = str(uid)
    uid_to_prot[uid] = gene
    if "," in uid:
        uids = uid.split(",")
        map[gene] = uid_map[uids[0]]
    else:
        try:
            map[gene] = uid_map[uid]
        except KeyError:
            print(f"Could not find {uid}")
            map[gene] = ""

import json

json.dump(map, open("gene_to_prot.json", "w"), indent=4)
with open("uid_to_prot.json", "w") as f:
    json.dump(uid_to_prot, f, indent=4)
"""
Last edited: 08-26-23
Author: Albert Cao
Description: Read protein data and find gene sequences
"""

import pandas as pd
import json

prot_levels = pd.read_csv('normal_tissue.tsv', sep='\t')
hpa = pd.read_csv('proteinatlas.tsv', sep='\t')
uniprot = pd.read_csv('uniprotkb_reviewed_true_AND_model_organ_2023_08_23.tsv', sep='\t')

uid_map = dict()
for ind in uniprot.index:
    uid = uniprot['Entry'][ind]
    seq = uniprot['Sequence'][ind]
    uid_map[uid] = seq

hpa_to_uid = dict()
for ind in hpa.index:
    gene = hpa['Gene'][ind]
    uid = hpa['Uniprot'][ind]
    if pd.isna(uid):
        continue
    uid = str(uid)
    if "," in uid:
        uids = uid.split(",")
        hpa_to_uid[gene] = uids[0]
    else:
        try:
            hpa_to_uid[gene] = uid
        except KeyError:
            print(f"Could not find {uid}")

hpa_map = dict()
genes = prot_levels['Gene name'].unique()
for gene in genes:
    try:
        uid = hpa_to_uid[gene]
        try:
            seq = uid_map[uid]
            if pd.isna(seq) or seq == "":
                print(f"sequence of {uid} is empty!")
            else:
                hpa_map[gene] = seq
        except:
            print(f"Could not find sequence of Uniprot ID {uid}")
    except:
        print(f"Could not find Uniprot ID of gene {gene}")

json.dump(hpa_map, open("hpa_gene_seqs.json", "w"), indent=4)

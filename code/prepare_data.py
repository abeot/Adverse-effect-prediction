"""
Last edited: 09-15-23
Author: Albert Cao
Description: Prepare drug data and combine it with protein data
"""

# %% [markdown]
# ## Process the drug SMILES info
# We only want the drugs for which we have frequency information (refer to [this paper](https://www.nature.com/articles/s41467-020-18305-y)). CID is given, and it's easy to get SMILES from that.
# 

# %%
import pandas as pd

cid_df = pd.read_csv('drug_cid.tsv', sep='\t')
cid_df
cid = dict()
for ind in cid_df.index:
    cid[cid_df['GenericName'][ind]] = cid_df['CID'][ind]
print(cid)
drugs = list(cid.keys())
print(drugs)
print(len(drugs))

# %% [markdown]
# ### Export drug CIDs for conversion (only first time use)

# %%
# Export the drug CIDs for conversion using this website: https://pubchem.ncbi.nlm.nih.gov/idexchange/
# Note that the total is 759
with open("drug_cid.txt", "w") as f:
    for drug in cid:
        f.write(str(cid[drug])+"\n")

# %% [markdown]
# ### Get drug SMILES dictionary

# %%
import pubchempy as pcp

drug_smiles = dict()
cid_smiles = dict()
with open("cid_smiles.txt", "r") as f:
    with open("cid_smiles_final.txt", "w") as o:
        for line in f.readlines():
            words = line.split()
            words[0] = int(words[0])
            if len(words) == 2:
                cid_smiles[words[0]] = words[1]
            else:
                c = pcp.Compound.from_cid(words[0])
                cid_smiles[words[0]] = c.isomeric_smiles
            o.write(f"{words[0]}\t{cid_smiles[words[0]]}\n")
for drug in drugs:
    drug_smiles[drug] = cid_smiles[cid[drug]]
print(len(drug_smiles))
print(type(drug_smiles))

# %% [markdown]
# ## Get the protein info (IC50s)

# %%
import json

f = open("hpa_gene_seqs.json")
prot_seqs: dict = json.load(f)
f.close()
prots = list(prot_seqs.keys())
print(len(prots))

# %% [markdown]
# ### Create the pairs for the prediction file

# %%
import pandas as pd
from tqdm import tqdm

def create_query_df(drugs: list, prots: list, smiles: dict, seq: dict):
    druglist = list()
    protlist = list()
    smileslist = list()
    seqlist = list()
    for drug in drugs:
        for prot in prots:
            druglist.append(drug)
            protlist.append(prot)
            smileslist.append(smiles[drug])
            seqlist.append(seq[prot])
    return pd.DataFrame.from_dict({
        "proteinID": protlist,
        "moleculeID": druglist,
        "proteinSequence": seqlist,
        "moleculeSmiles": smileslist,
    })
def to_tsv_tqdm(df: pd.DataFrame, filename: str):
    with open(filename, "w") as f:
        f.write("proteinID\tmoleculeID\tproteinSequence\tmoleculeSmiles\n")
        for ind in tqdm(df.index, total=df.shape[0], desc='Exporting df'):
            prot = df['proteinID'][ind]
            drug = df['moleculeID'][ind]
            seq = df['proteinSequence'][ind]
            smiles = df['moleculeSmiles'][ind]
            f.write(f"{prot}\t{drug}\t{seq}\t{smiles}\n")


query_df = create_query_df(drugs, prots, drug_smiles, prot_seqs)
query_df 
# to_tsv_tqdm(query_df, 'query_df.tsv')

# %% [markdown]
# ### Use ConPLex to get IC50 values

# %%
# This code block uses ConPlex to predict binding affinity values
# code reference: https://github.com/samsledje/ConPLex
from conplex_dti.model.architectures import SimpleCoembeddingNoSigmoid
from conplex_dti.featurizer.protein import ProtBertFeaturizer
from conplex_dti.featurizer.molecule import MorganFeaturizer
import torch
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cpu")
print("Loading models")
target_featurizer = ProtBertFeaturizer(
    save_dir='.', per_tok=False
).cpu()
drug_featurizer = MorganFeaturizer(save_dir='.').cpu()

drug_featurizer.preload(query_df["moleculeSmiles"].unique())
target_featurizer.preload(query_df["proteinSequence"].unique())

model = SimpleCoembeddingNoSigmoid(
    drug_featurizer.shape, target_featurizer.shape, 
    latent_dimension=1024,
    latent_activation="GELU",
    latent_distance="Cosine",
    classify=False
)
model_path = './models/Affinity82923_best_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()
model = model.to(device)

dt_feature_pairs = [
        (drug_featurizer(r["moleculeSmiles"]), target_featurizer(r["proteinSequence"]))
        for _, r in tqdm(query_df.iterrows(), desc="Calculating feature pairs", total=query_df.shape[0])
    ]
dloader = DataLoader(dt_feature_pairs, batch_size=32, shuffle=False)

print(f"Generating predictions...")
preds = []
with torch.set_grad_enabled(False):
    for b in tqdm(dloader, desc="Calculating predictions"):
        preds.append(model(b[0], b[1]).detach().cpu().numpy())

preds = np.concatenate(preds)

result_df = pd.DataFrame(query_df[["moleculeID", "proteinID"]])
result_df["Prediction"] = preds

print(f"Printing ConPLex results to drug_prot_ic50_v2.tsv")
result_df.to_csv('drug_prot_ic50_v2.tsv', sep="\t", index=False, header=False)

# %% [markdown]
# ## Get the frequency info and tissue expression levels

# %%
import pandas as pd 
from tqdm import tqdm

freq_df = pd.read_csv('all_grades.tsv', sep='\t')
reactions = freq_df['reaction'].unique().tolist()

drug_reaction = dict()

for drug in drugs:
    drug_reaction[drug] = dict()
    for reaction in reactions:
        drug_reaction[drug][reaction] = -100

for ind in tqdm(freq_df.index):
    drug = freq_df['GenericName'][ind]
    reaction = freq_df['reaction'][ind]
    label = freq_df['label'][ind]
    drug_reaction[drug][reaction] = int(label)

import json

with open("drug_reaction_freq.json", "w") as f:
    json.dump(drug_reaction, f)

# %%
import pandas as pd
from tqdm import tqdm

hpa_df = pd.read_csv('normal_tissue.tsv', sep='\t')
tissues = hpa_df['Tissue'].unique().tolist()
prot_levels = dict()

for prot in prots:
    prot_levels[prot] = dict()
    for tissue in tissues:
        prot_levels[prot][tissue] = -100

for ind in tqdm(hpa_df.index, desc="Generating expression level info"):
    prot = hpa_df['Gene name'][ind]
    tissue = hpa_df['Tissue'][ind]
    level = hpa_df['Level'][ind]
    if level == 'High':
        level = 3
    elif level == 'Medium':
        level = 2
    elif level == 'Low':
        level = 1
    else:
        level = 0
    if prot in prots:
        if prot_levels[prot][tissue] == -100:
            prot_levels[prot][tissue] = level 
        else:
            prot_levels[prot][tissue] += level

import json

with open("prot_tissue_levels.json", "w") as f:
    json.dump(prot_levels, f, indent=4)

# %% [markdown]
# ## Reformat the files
# - drug_smiles - dictionary mapping drug to its smiles
# 
# - result_df - the dataframe with drug and target IC50
# 
# - drug_reaction - dictionary mapping drug and reaction to frequency class (or -100 for na value)
# 
# - prot_levels - dictionary mapping protein and tissue to expression level (3 2 1 0 for high medium low not detected or -100 for na value)

# %% [markdown]
# ### Drug, SMILES and IC50

# %%
drug_smiles_ic50_dict = dict()

drug_ic50s = dict()
for drug in drugs:
    drug_ic50s[drug] = dict()

for ind in tqdm(result_df.index, desc="Generate drug IC50 dict"):
    drug = result_df['moleculeID'][ind]
    prot = result_df['proteinID'][ind]
    ic50 = result_df['Prediction'][ind]
    drug_ic50s[drug][prot] = ic50
    

with open("drug_smiles_ic50.tsv", "w") as f:
    f.write("drug\tSMILES\t")
    for prot in prots:
        f.write(f"{prot}\t")
    f.write("\n")
    for drug in tqdm(drugs, desc="Generate file for drug, SMILES and IC50"):
        smiles = drug_smiles[drug]
        ic50s = drug_ic50s[drug]
        f.write(f"{drug}\t{smiles}\t")
        for prot in prots:
            f.write(f"{ic50s[prot]}\t")
        f.write("\n")


# %% [markdown]
# ### Drug and reaction frequency

# %%
with open("drug_reaction_freq.tsv", "w") as f:
    f.write("drug\t")
    for reaction in reactions:
        f.write(f"{reaction}\t")
    f.write("\n")
    for drug in tqdm(drugs, desc="Generate file for drug and reaction frequency"):
        f.write(f"{drug}\t")
        for reaction in reactions:
            f.write(f"{drug_reaction[drug][reaction]}\t")
        f.write("\n")

# %% [markdown]
# ### Protein and tissue expression levels

# %%
with open("prot_tissue_levels.tsv", "w") as f:
    f.write("protein\t")
    for tissue in tissues:
        f.write(f"{tissue}\t")
    f.write("\n")
    for prot in tqdm(prots, desc="Generate file for protein tissue expression levels"):
        f.write(f"{prot}\t")
        for tissue in tissues:
            f.write(f"{prot_levels[prot][tissue]}\t")
        f.write("\n")



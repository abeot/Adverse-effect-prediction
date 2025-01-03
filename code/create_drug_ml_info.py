"""
Last edited: 09-15-23
Author: Albert Cao
Description: collect data into one file for model training
"""

# Tasks done (in order):
# - Get drug reaction info (only look at top 5 reactions with most data)   
# - Get protein cell level info   
# - Get the IC50 information from TDC and ConPLex
# - Combine the dataframes using multiplication  
# 
# These tasks are done in a separate file:
# - Get MACCS fingerprint  
# - Create the train test split and dataloaders  
# - Create the models  
# - Train the model  
# - Make some predictions  

# # Data collection
# ## Get the drug info
# ### Get our reaction data

import pandas as pd
import json
from tdc.multi_pred import DTI
from tqdm import tqdm

from conplex_dti.model.architectures import SimpleCoembeddingNoSigmoid
from conplex_dti.featurizer.protein import ProtBertFeaturizer
from conplex_dti.featurizer.molecule import MorganFeaturizer
import torch
from torch.utils.data import DataLoader
import numpy as np

reaction_df = pd.read_table('drug_reaction_freq.tsv')
reaction_df = reaction_df[['drug', 'diarrhoea', 'headache', 
                           'nausea', 'vomiting', 'dizziness']]

start_col = reaction_df.columns.get_loc('diarrhoea')
end_col = reaction_df.columns.get_loc('dizziness')
for i in range(start_col, end_col+1):
    col = reaction_df.columns[i]
    a = list(reaction_df[col] != -100)
    print(a.count(True))


# ### Get the Pubchem IDs (needed for IC50)
cid_df = pd.read_table('drug_cid.tsv')
drug_to_cid = dict()
for ind in cid_df.index:
    drug_to_cid[cid_df['GenericName'][ind]] = cid_df['CID'][ind]
print(len(drug_to_cid.keys()))
cids = set(drug_to_cid.values())
print(cids)


# ## Get the protein info
# ### Get tissue expression level
with open("prot_cell_levels.json") as f:
    cell_level = json.load(f)
prots = list(cell_level.keys())
print(len(prots))

# ### Get the IC50 values
ic_df1 = pd.read_csv('train_val.csv')
ic_df2 = pd.read_csv('test.csv')
ic_df = pd.concat([ic_df1, ic_df2], ignore_index=True)
ic_df = ic_df.drop(columns=['Year'])
ic_df = ic_df[['Drug_ID', 'Target_ID', 'Target', 'Y']]
print(f"unique proteins (in Uniprot ID): {len(ic_df['Target_ID'].unique())}")
print(f"unique drugs (in Pubchem ID): {len(ic_df['Drug_ID'].unique())}")
uniprot_ids = set(ic_df['Target_ID'].unique())

ic_df_subset = ic_df[ic_df['Drug_ID'].isin(cids)]
print(f"unique drugs that also have side effect info: {len(ic_df_subset['Drug_ID'].unique())}") 
## This value is too small, use the full dataset from TDC

data = DTI(name = "BindingDB_IC50")
data.harmonize_affinities(mode = "mean")
data.convert_to_log(form = "binding")
split = data.get_split()
full_df = pd.concat([split['train'], split['test'], split['valid']], ignore_index=True)
full_df = full_df[full_df['Drug_ID'].isin(cids)]
full_df = full_df[full_df['Target_ID'].isin(uniprot_ids)]
full_df = full_df.drop(columns=['Target'])
print(f"unique drugs that also have side effect data: {len(full_df['Drug_ID'].unique())}") # Value is also quite low

affinities = dict()

known_dti_pairs = set()
for ind in full_df.index:
    drug_cid = full_df['Drug_ID'][ind]
    prot_uid = full_df['Target_ID'][ind]
    known_dti_pairs.add((drug_cid, prot_uid))
    affinity = full_df['Y'][ind]
    affinities[(drug_cid, prot_uid)] = affinity

# ### Using ConPLex to predict IC50 values
# create query df (this doesn't need to run after the first time)
cid_to_smiles = {}
with open("cid_smiles_final.txt") as f:
    for line in f.readlines():
        words = line.split()
        cid = int(words[0])
        cid_to_smiles[cid] = words[1]
uid_to_sequence = {}
for ind in ic_df.index:
    uid = ic_df['Target_ID'][ind]
    seq = ic_df['Target'][ind]
    uid_to_sequence[uid] = seq
print(f"drugs: {len(cid_to_smiles.keys())}")
print(f"prots: {len(uid_to_sequence.keys())}")

query_df = {"proteinID": list(),
            "moleculeID": list(),
            "proteinSequence": list(),
            "moleculeSmiles": list()
}
for cid in cids:
    for uid in uniprot_ids:
        if (cid, uid) in known_dti_pairs:
            continue 
        smiles = cid_to_smiles[cid]
        sequence = uid_to_sequence[uid]
        query_df['proteinID'].append(uid)
        query_df['moleculeID'].append(cid)
        query_df['proteinSequence'].append(sequence)
        query_df['moleculeSmiles'].append(smiles)
query_df = pd.DataFrame.from_dict(query_df)
# query_df.to_csv('query_df.tsv', sep='\t', index=False)

# Use ConPLex to predict the other pIC50 vals
# code reference: 
# https://github.com/samsledje/ConPLex
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
result_df.to_csv('query_df_results.tsv', index=False)


# ### Read IC50 values
result_df = pd.read_csv('query_df_results.csv')
for ind in result_df.index:
    cid = result_df['moleculeID'][ind]
    uid = result_df['proteinID'][ind]
    affinity = result_df['Prediction'][ind]
    affinities[(cid, uid)] = affinity

drug_prot_ic = {}
with open("uid_to_prot.json") as f:
    uniprot_to_prot = json.load(f)
cid_to_drug = {}
for drug in drug_to_cid.keys():
    cid_to_drug[drug_to_cid[drug]] = drug

for cid in cids:
    drug = cid_to_drug[cid]
    drug_prot_ic[drug] = {}
    for uid in uniprot_ids:
        affinity = affinities[(cid, uid)]
        prot = uniprot_to_prot[uid]
        drug_prot_ic[drug][prot] = affinity
# Just to make sure
print(f"total distinct proteins: {len(drug_prot_ic['alfentanil'].keys())}")
print(f"total distinct drugs: {len(drug_prot_ic.keys())}")

with open("drug_prot_ic.json", "w") as f:
    json.dump(drug_prot_ic, f, indent=4)

# Export to a dataframe
drugs = set(drug_prot_ic.keys())
prots = set(drug_prot_ic['alfentanil'].keys())

drug_ic_df = pd.read_csv('drug_smiles.csv')
for prot in prots:
    drug_ic_df[prot] = drug_ic_df['drug'].apply(lambda x : drug_prot_ic[x][prot])
drug_ic_df.to_csv('drug_smiles_ic50.csv', index=False)

# # Data cleaning
# Dataframes that are the inputs:
# - reaction_df
# - drug_ic_df
# - cell_level
# 
# Miscellaneous stuff that should be correct:
# - drugs
# - drug_prot_ic
# - prots

# ## Get cell 'strength' and prepare data for ML
from tqdm import tqdm

cells = set(cell_level['PLCL2'].keys())
cells.remove("nan")
print(len(cells))
drug_strength = {}
drug_strength['drug'] = list()
drug_strength['SMILES'] = list()
for cell in cells:
    drug_strength[cell] = list()

# Get the proteins that have cell level info
correct_prots = set()
for prot in prots:
    if prot in cell_level.keys():
        correct_prots.add(prot)
print(f"proteins that have expression level and ic info: {len(correct_prots)}")
prots = correct_prots

for ind in tqdm(drug_ic_df.index, total=drug_ic_df.shape[0], desc='Calculate cell strength'):
    drug = drug_ic_df['drug'][ind]
    drug_strength['drug'].append(drug)
    smiles = drug_ic_df['SMILES'][ind]
    drug_strength['SMILES'].append(smiles)
    for cell in cells:
        total = -100
        for prot in prots:
            ic = drug_ic_df[prot][ind]
            level = cell_level[prot][cell]
            ans = -100
            try:
                if level >= 0:
                    ans = level * ic
            except:
                print(f"level not found for {prot} and cell {cell}")
            if ans != -100:
                if total == -100:
                    total = ans
                else:
                    total += ans
        drug_strength[cell].append(total)
drug_strength = pd.DataFrame.from_dict(drug_strength) 
full_df = drug_strength.merge(reaction_df)
full_df.to_csv('drug_ml_info.csv', index=False)



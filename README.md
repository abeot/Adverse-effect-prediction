# Adverse-effect-prediction

This is the source code for [Machine Learning Prediction of On/Off Target-driven Clinical Adverse Events](https://doi.org/10.1007/s11095-024-03742-x).

The code for the final model we used for predictions is contained in `AE_train_1.ipynb`.
This code was original work.

The relevant cleaned data is found in the `data_normalized_X.csv` where `X` is one of the 
five adverse effects diarrhoea, dizziness, headache, nausea and vomiting.
These files were cleaned by us.

The bulk of the code that cleans and processses the data is contained in the 
`data_processing` folder. All the code written in the folder is original.

The original data comes from the files `all_grades.tsv`, `drug_cid.tsv`, `normal_tissue.tsv`,
`proteinatlas.tsv`, `uniprotkb_reviewed_true_AND_model_organ_2023_08_23.tsv`. 

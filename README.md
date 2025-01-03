# Adverse-effect-prediction

This is the source code for [Machine Learning Prediction of On/Off Target-driven Clinical Adverse Events](https://doi.org/10.1007/s11095-024-03742-x).

![Alt Text](overview.png)


The code for the final model we used for predictions is contained in `AE_train_1.py` in the `code` folder.
A.C. and Y.Z. worked together on `AE_train_1.py`, `utils.py`, and `ml_utils.py`. 

The original data comes from the files `all_grades.tsv`, `drug_cid.tsv`, `normal_tissue.tsv`,
`proteinatlas.tsv`, `uniprotkb_reviewed_true_AND_model_organ_2023_08_23.tsv`. These
datasets were acquired from various sources referenced in the paper.


<details><summary><b>Citation</b></summary>

If you use this code or the models in your research, please cite the following paper:

```bibtex
@article{cao2024machine,
  title={Machine Learning Prediction of On/Off Target-driven Clinical Adverse Events},
  author={Cao, Albert and Zhang, Luchen and Bu, Yingzi and Sun, Duxin},
  journal={Pharmaceutical Research},
  volume={41},
  number={8},
  pages={1649--1658},
  year={2024},
  publisher={Springer}
}
```

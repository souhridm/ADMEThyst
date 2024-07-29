# ADMEThyst: Drug-induced cardiotoxicity (DICT) prediction using ADMET-AI

ADMEThyst is a tool for predicting Drug-Induced Cardiotoxicity (DICT) using an Extreme Gradient Boosting (XGB) model based on 41 ADMET properties predicted by [ADMET-AI](https://github.com/swansonk14/admet_ai). ADMEThyst was trained on drugs from the [DICTrank dataset](https://doi.org/10.1016/j.drudis.2023.103770). ADMEThyst was developed jointly by Stanford University and [Greenstone Biosciences](https://greenstonebio.com/).


## Installation

Optionally, first create a new conda environment and activate it:

```bash
conda create -n admethyst python=3.11
conda activate admethyst
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```


## Usage

Run `scripts/Train_&_Plot_ADMET_AI_DICTrank_predictor.ipynb` to train using 555 drugs (262 no DICT concern, 293 most DICT concern) from DICTrank using 41 ADMET properties from ADMET-AI. This notebook also plots performance (ROC AUC and PR AUC curves), compares ADMET-AI with SwissADME, identifies most important features (SHAP), and plots radial plots for least toxic and most toxic cardiovascular drugs.

Run `scripts/Predict_ADMET_AI_DICTrank.py` to get 41 ADMET property predictions from ADMET-AI and DICTrank predictions for SMILES string(s) or a list of SMILES strings in a `.txt` file.

Example usage.
```bash
python scripts/Predict_ADMET_AI_DICTrank.py \
    --smiles "O=C(OC1CC2CC3CC(C1)[NH+]2CC3=O)c1c[nH]c2ccccc12" \
             "O=c1n(CCC[NH+]2CCN(c3cccc(Cl)c3)CC2)nc2ccccn12" \
             "CC(C)COc1ccc(CNC(=O)[NH+](Cc2ccc(F)cc2)C2CC[NH+](C)CC2)cc1" \
             "C[NH+](C)C([NH3+])=NC(N)=[NH2+]" \
             "CC(C(=O)[O-])c1ccc(-c2ccccc2)c(F)c1" \
    --name admet_ai_dictrank \
    --out_dir test_prediction
```

Run `python scripts/Predict_ADMET_AI_DICTrank.py -h` for more information.

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

Run `scripts/Predict_ADMET_AI_DICTrank.py` to get 41 ADMET property predictions from ADMET-AI and DICTrank predictions for an individual SMILES string or a list of SMILES strings in a `.txt` file.

```bash
python scripts/Predict_ADMET_AI_DICTrank.py -h                                                          
    usage: Predict_ADMET_AI_DICTrank.py [-h] [-s SMILES] [-l LIST] [-n NAME] [-o OUT_PATH]
    
    options:
      -h, --help            show this help message and exit
      -s SMILES, --smiles SMILES
                            individual SMILES string, to predict DICT concern - based on ADMET-AI
      -l LIST, --list LIST  .txt file contining list of SMILES strings, to predict DICT concern - based on
                            ADMET-AI
      -n NAME, --name NAME  Run ID or name
      -o OUT_PATH, --out_path OUT_PATH
                          Output folder or directory
```

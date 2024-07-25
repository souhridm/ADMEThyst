# ADMEThyst
## DICT concern prediction (concern for Drug induced cardio-toxicity)
Drug induced toxicity predictions at organ and tissue level, trained on 41 ADMET predictions from ADMET-AI (https://github.com/swansonk14/admet_ai.git)
training data : Drugs obtained from DICTrank database (https://www.sciencedirect.com/science/article/pii/S1359644623002866, 
                                                       https://www.biorxiv.org/content/10.1101/2023.10.15.562398v1, 
                                                       https://www.fda.gov/science-research/bioinformatics-tools/drug-induced-cardiotoxicity-rank-dictrank-dataset
                                                      )

Requirements:

- chemprop (https://github.com/chemprop/chemprop.git) (chemprop==1.6.1)
- ADMET-AI (https://github.com/swansonk14/admet_ai.git) (admet-ai==1.2.0)
- pandas (pandas==1.5.3)
- numpy (numpy==1.24.1)
- os
- matplotlib (matplotlib==3.6.3)
- seaborn(seaborn==0.12.2)
- sklearn (sklearn==0.0.post1_
- shap (shap==0.45.0)
- _pickle
- scipy (scipy==1.13.0)

Steps:

1. Install admet-ai and chemprop
2. Open scripts/Train_&_Plot_ADMET_AI_DICTrank_predictor.ipynb to train using 555 drugs (262 no DICT concern, 293 most DICT concern) from DICTrank using 41 ADMET properties from ADMET-AI, plot performance (ROC, PR curves), comparison with SwissADME, identify most important features (SHAP) and plot radial plots for least toxic and most toxic cardiovascular drugs.
3. Run Predict_ADMET_AI_DICTrank.py
   python /Users/souhridm/Downloads/Predict_ADMET_AI_DICTrank.py -h                                                          

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








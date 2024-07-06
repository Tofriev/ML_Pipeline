# %%
import pandas as pd 



# set parameters
parameters = {
    "random_state": 42,
    "train_size": 0.8,
    "Dataset": {
        "target": "mortality",
        "numerical_features": [
        "LOS", "Age", "Weight", "Height", "Bmi", "Temp", "RR", "HR", "GLU", "SBP", "DBP", "MBP", "Ph", "GCST", "PaO2", 
        "Kreatinin", "FiO2", "Kalium", "Natrium", "Leukocyten", "Thrombocyten", "Bilirubin", "HCO3", "Hb", "Quick", 
        "ALAT", "ASAT", "PaCO2", "Albumin", "AnionGAP"
    ],
        },
    "Trainer": {
        "models": ["LogReg", "EBM", "XGB"],
        "hpo": "grid_params.json",
        "cv_folds": 5,
    }
}

# load data
with open("data/mimic4_mean_100_extended_filtered.csv") as file: 
    data = pd.read_csv(file)








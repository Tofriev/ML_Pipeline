# %%

import sys
import os
import json
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)
import pandas as pd 
from src.Pipeline import Pipeline

with open("../src/hpo.json") as file:
    hpo = json.load(file)
# set parameters
parameters = {
    "random_state": 42,
    "Dataset": {
        "train_size": 0.8,
        "target": "mortality",
        "numerical_features": 
        [
            "Age", "Temp", "RR", "HR", "GLU", "MBP", "Ph", "GCST", "PaO2", 
            "Kreatinin", "FiO2", "Kalium", "Natrium", "Leukocyten", "Thrombocyten", "Bilirubin", "HCO3", "Hb", "Quick",
            "PaCO2", "Albumin", "AnionGAP"
        ],
        "categorical_features":
        [
            "Eth", "gender"
        ],
        
        "imputation": "knn",
        "sampling": "smote"
        }, 
    "Trainer": {
        "models": ["LogReg", "EBM", "XGB"], 
        "hpo": hpo,
        "cv_folds": 5,
    }
}

# load data
with open("../data/mimic4_total_new.csv") as file: 
    data = pd.read_csv(file)

pipe = Pipeline(parameters, data)
pipe.run()
results = pipe.return_results()
print(results)








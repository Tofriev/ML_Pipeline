#%%
import pandas as pd

with open("../data/mimic4_mean_100_extended_filtered.csv") as file: 
    data_old = pd.read_csv(file)
with open("../data/mimic4_total.csv") as file: 
    data_new = pd.read_csv(file)
with open("../data/mimic4_total_new.csv") as file: 
    data_newnew = pd.read_csv(file)



    missing_values_old = data_old.isnull().sum()
    missing_values_new = data_new.isnull().sum()
    missing_values_newnew = data_newnew.isnull().sum()

    print("Missing values in old data:")
    print(missing_values_old)

    print("Missing values in new data:")
    print(missing_values_new)

    print("Missing values in newnew data:")
    print(missing_values_newnew)


    
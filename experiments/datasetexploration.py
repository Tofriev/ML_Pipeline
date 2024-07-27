#%%              
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = "../data/mimic4_finalfinalfinal.csv"

numerical_vars = [
            "Age", "Weight", "Height", "Temp", "RR", "HR", "GLU", "MBP", "Ph", "GCST", "PaO2", 
            "Kreatinin", "FiO2", "Kalium", "Natrium", "Leukocyten", "Thrombocyten", "Bilirubin", "HCO3", "Hb", "Quick",
            "PaCO2", "ALAT", "ASAT", "Albumin", "AnionGAP", "Lactate", "Harnstoff"
        ]
categorical_vars = ["Eth", "gender"]
target_var = "mortality"

df = pd.read_csv(csv_file_path)

df = df.dropna(thresh=df.shape[1]*0.5)

# Histograms
for var in numerical_vars:
    plt.figure()
    df[var].hist()
    plt.title(f"Histogram for {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.show()

# Bar Plots
for var in categorical_vars:
    plt.figure()
    df[var].value_counts().plot(kind="bar")
    plt.title(f"Bar Plot for {var}")
    plt.xlabel(var)
    plt.ylabel("Count")
    plt.show()

# Summary Table
total_rows = len(df)
summary_data = []

for var in numerical_vars:
    missing_values = df[var].isnull().sum()
    missing_percentage = (missing_values / total_rows) * 100
    mean = df[var].mean()
    std = df[var].std()
    summary_data.append([var, "Numerical", f"{missing_percentage:.2f}%", mean, std, None])

for var in categorical_vars:
    missing_values = df[var].isnull().sum()
    missing_percentage = (missing_values / total_rows) * 100
    percentage = df[var].value_counts(normalize=True) * 100
    percentage_str = ", ".join([f"{k}: {v:.2f}%" for k, v in percentage.items()])
    summary_data.append([var, "Categorical", f"{missing_percentage:.2f}%", None, None, percentage_str])

summary_df = pd.DataFrame(summary_data, columns=["Variable", "Type", "Missing Values (%)", "Mean", "Std", "Category Percentages"])
print(summary_df)

# Percentages Target
target_percentage = df[target_var].value_counts(normalize=True) * 100
print(f"\nPercentage of the binary target variable:\n{target_percentage}")

# Correlation Matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_vars].corr(), annot=True, fmt=".2f")
plt.title("Correlation Matrix for Numerical Variables")
plt.show()

# Box Plots Against Target
for var in numerical_vars:
    plt.figure()
    sns.boxplot(x=target_var, y=var, data=df)
    plt.title(f"Box Plot of {var} by {target_var}")
    plt.show()

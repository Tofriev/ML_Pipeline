#%%
import json
import pandas as pd
from matplotlib import pyplot as plt

with open("results_fracs.json") as file:
    data = json.load(file)

records = []
for data_amount, metrics in data.items():
    for metric in metrics:
        records.append({
            "data": int(data_amount),
            "model": metric["model"],
            "roc_auc": metric["roc_auc"],
            "auroc_std": metric["auroc_std"]
        })

df = pd.DataFrame(records)

y_min = df["roc_auc"].min() - df["auroc_std"].max()
y_max = df["roc_auc"].max() + df["auroc_std"].max()


plt.figure(figsize=(18, 6))

models = df['model'].unique()
line_styles = {"LogReg": "dotted", "EBM": "dashed", "XGB": "solid"}
colors = {"LogReg": "blue", "EBM": "green", "XGB": "red"}

for i, model in enumerate(models, 1):
    plt.subplot(1, 3, i)
    model_data = df[df['model'] == model]
    plt.errorbar(
        model_data["data"], 
        model_data["roc_auc"], 
        yerr=model_data["auroc_std"], 
        label=model, 
        linestyle=line_styles[model], 
        color=colors[model],
        capsize=5
    )
    plt.xlabel("Number of Data Samples")
    plt.ylabel("ROC AUC")
    plt.title(f"ROC AUC vs N of Data Samples for {model} ")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    for x in range(0, 70001, 10000):
        plt.annotate(f"{x}", xy=(x, 0.5), xytext=(x, 0.5), textcoords="data", ha="center")

plt.tight_layout()
plt.show()

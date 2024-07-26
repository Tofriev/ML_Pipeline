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
line_styles = {"LogReg": "solid", "EBM": "solid", "XGB": "solid"}
colors = {"LogReg": "blue", "EBM": "green", "XGB": "red"}

for i, model in enumerate(models, 1):
    plt.subplot(1, 3, i)
    model_data = df[df['model'] == model]
    x = model_data["data"]
    y = model_data["roc_auc"]
    yerr = model_data["auroc_std"]

    plt.plot(x, y, label=model, linestyle=line_styles[model], color=colors[model])
    plt.fill_between(x, y - yerr, y + yerr, color='grey', alpha=0.3)
    
    plt.xlabel("Number of Data Samples")
    plt.ylabel("ROC AUC")
    plt.title(f"ROC AUC vs N of Data Samples for {model}")
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    for x in range(0, 70001, 10000):
        plt.annotate(f"{x}", xy=(x, 0.5), xytext=(x, 0.5), textcoords="data", ha="center")

plt.tight_layout()
plt.show()

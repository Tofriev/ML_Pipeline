#%%
import json
import pandas as pd
from matplotlib import pyplot as plt

with open("results_fracs.json") as file:
    data = json.load(file)

records = []
for data_amount, metrics in data.items():
    for metric in metrics:
        records.append({"data": int(data_amount), "model": metric["model"], "roc_auc": metric["roc_auc"]})

df = pd.DataFrame(records)

plt.figure(figsize=(12, 8))
line_styles = {"LogReg": "dotted", "EBM": "dashed", "XGB": "solid"}

for model, group_data in df.groupby("model"):
    plt.plot(group_data["data"], group_data["roc_auc"], label=model, linestyle=line_styles[model])

plt.xlabel("Number of Data Samples")
plt.ylabel("ROC AUC")
plt.title("ROC AUC vs Number of Data Samples for Different Models")
plt.legend()
plt.grid(True)

for x in range(0, 70001, 10000):
    plt.annotate(f"{x}", xy=(x, 0.5), xytext=(x, 0.5), textcoords="data", ha="center")

plt.show()
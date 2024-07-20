#%%
import pandas as pd
import matplotlib.pyplot as plt
import json

with open('results.json') as file:
    data = json.load(file)

df = pd.DataFrame(data).T

df_transposed_filtered = df[['roc_auc', 'accuracy', 'recall', 'precision']].T

fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.25
r1 = range(len(df_transposed_filtered.index))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

bars1 = ax.barh(r1, df_transposed_filtered['LogReg'], height=bar_width, label='LogReg', color='blue')
bars2 = ax.barh(r2, df_transposed_filtered['EBM'], height=bar_width, label='EBM', color='green')
bars3 = ax.barh(r3, df_transposed_filtered['XGB'], height=bar_width, label='XGB', color='orange')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', 
                va='center', ha='left', color='black', fontsize=10)

ax.set_yticks([r + bar_width for r in range(len(df_transposed_filtered.index))])
ax.set_yticklabels(df_transposed_filtered.index)
ax.set_xlabel('Values')
ax.set_title('Model Comparison (ROC AUC, Accuracy, Recall, Precision)')
ax.invert_yaxis() 
ax.legend()

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Load the data
# ======================================================================================================================
dm_data = pd.read_csv('./data/dm_data.csv')
img_data = pd.read_csv('./data/img_data.csv')

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=dm_data, x='Block', y='BestOption', hue='Condition', errorbar='ci')
plt.title('Best Option Selection by Block and Condition')
plt.xlabel('Block Number')
plt.ylabel('Proportion of Best Option Selected')
plt.xticks(np.arange(0, 20, 2))
plt.savefig('./figures/BestOptionByBlock.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=dm_data, x='Block', y='HighFreqOption', hue='Condition', errorbar='ci')
plt.title('Best Option Selection by Block and Condition')
plt.xlabel('Block Number')
plt.ylabel('Proportion of Best Option Selected')
plt.xticks(np.arange(0, 20, 2))
plt.savefig('./figures/HighFreqOptionByBlock.png', dpi=300)
plt.show()


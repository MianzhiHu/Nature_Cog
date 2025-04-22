import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Load the data
# ======================================================================================================================
dm_data = pd.read_csv('./data/dm_data.csv')
img_data = pd.read_csv('./data/img_data.csv')
stimuli_info = pd.read_csv('./stimuli/stimuli_info.csv')

# Add all stimuli info to the image data
stimuli_info.rename(columns={'ImageName': 'image_name'}, inplace=True)
img_data = img_data.merge(stimuli_info, on='image_name', how='left')
avg_rating = img_data.groupby(['Subnum']).agg({
    'Condition': 'first',
    'Gender': 'first',
    'Ethnicity': 'first',
    'Race': 'first',
    'Age': 'first',
    'naturalness': 'mean',
    'disorderliness': 'mean',
    'aesthetic': 'mean',
    'Hue': 'mean',
    'Bright': 'mean',
    'Saturaton': 'mean',
    'SDhue': 'mean',
    'SDsat': 'mean',
    'Sdbright': 'mean',
    'Entropy': 'mean',
    'Perc_Nat': 'mean',
    'pc1f': 'mean',
    'pc2f': 'mean',
    'pc3f': 'mean',
    'pc4f': 'mean',
    'SED': 'mean',
    'total_ED': 'mean',
    'NSED': 'mean'
}).reset_index()
avg_rating.to_csv('./data/avg_rating.csv')

# ======================================================================================================================
# Statistical Analysis
# ======================================================================================================================
dm_summary = dm_data.groupby(['Subnum', 'Condition', 'Task', 'Block']).agg({
    'BestOption': 'mean',
    'HighFreqOption': 'mean'
}).reset_index()
dm_summary['Condition'] = pd.Categorical(dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
dm_summary = dm_summary.merge(avg_rating, on=['Subnum', 'Condition'], how='left')
dm_summary_IGT = dm_summary[dm_summary['Task'] == 'IGT']
dm_summary_SGT = dm_summary[dm_summary['Task'] == 'SGT']
dm_summary_IGT.to_csv('./data/dm_summary_IGT.csv')

# # Create the plot
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=dm_data, x='Block', y='BestOption', hue='Condition', errorbar='ci')
# plt.title('Best Option Selection by Block and Condition')
# plt.xlabel('Block Number')
# plt.ylabel('Proportion of Best Option Selected')
# plt.xticks(np.arange(0, 20, 2))
# plt.savefig('./figures/BestOptionByBlock.png', dpi=300)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=dm_data, x='Block', y='HighFreqOption', hue='Condition', errorbar='ci')
# plt.title('Best Option Selection by Block and Condition')
# plt.xlabel('Block Number')
# plt.ylabel('Proportion of Best Option Selected')
# plt.xticks(np.arange(0, 20, 2))
# plt.savefig('./figures/HighFreqOptionByBlock.png', dpi=300)
# plt.show()
#

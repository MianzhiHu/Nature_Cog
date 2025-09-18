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
avg_rating = pd.read_csv('./data/avg_rating.csv')

dm_data['Condition'] = pd.Categorical(dm_data['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
task_2nd = dm_data[dm_data['TaskCode'] == 2]
print(f'The number of participants who did SGT second: {task_2nd["Subnum"].nunique()}')
print(f'Conditions in SGT second: {task_2nd["Condition"].value_counts() // 250}')

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

# Summary for overall performance
overall_summary = dm_data.groupby(['Subnum', 'Condition', 'Task'], observed=False).agg({
    'BestOption': 'mean',
    'HighFreqOption': 'mean',
}).dropna().reset_index()
overall_summary = overall_summary.pivot_table(index=['Subnum', 'Condition'], columns='Task', values=['BestOption', 'HighFreqOption'])
overall_summary.columns = ['_'.join(col).strip() for col in overall_summary.columns.values]
overall_summary = overall_summary.reset_index()
overall_summary.to_csv('./data/dm_summary_overall.csv', index=False)

# Calculate deck selection proportions
deck_counts = task_2nd.groupby(['Subnum', 'Condition', 'Task', 'keyResponse'], observed=True).size().reset_index(name='counts')
deck_counts['proportion'] = deck_counts['counts'] / 100

img_rating_summary = img_data.groupby(['image_name', 'Condition']).agg({
    'naturalness': 'mean',
    'disorderliness': 'mean',
    'aesthetic': 'mean',
    'Perc_Nat': 'mean'
}).reset_index()
# img_rating_summary = img_rating_summary[img_rating_summary['Condition'] != 'Control']

# run correlation analysis
print(pg.corr(img_rating_summary['naturalness'], img_rating_summary['Perc_Nat'], method='pearson'))
print(pg.corr(img_rating_summary['naturalness'], img_rating_summary['disorderliness'], method='pearson'))
print(pg.corr(img_rating_summary['naturalness'], img_rating_summary['aesthetic'], method='pearson'))
print(pg.corr(img_rating_summary['disorderliness'], img_rating_summary['aesthetic'], method='pearson'))

# mixed effects model
me_model = ols('disorderliness ~ aesthetic + C(Condition) + (1|Subnum)', data=img_data).fit()
print(me_model.summary())

# ======================================================================================================================
# Plotting
# ======================================================================================================================
# Create a correlation plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='naturalness', y='Perc_Nat', hue='Condition', data=img_rating_summary, alpha=0.5)
sns.regplot(x='naturalness', y='Perc_Nat', data=img_rating_summary, scatter=False,
            line_kws={'color': 'red', 'linewidth': 2})
plt.xlabel('Observed Naturalness Rating')
plt.ylabel('Original Naturalness Rating')
plt.legend(title='Condition')
sns.despine()
plt.tight_layout()
plt.savefig('./figures/Naturalness_vs_Original.png', dpi=600)
plt.show()

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=task_2nd, x='Block', y='BestOption', hue='Condition', errorbar='ci')
plt.xlabel('Block Number')
plt.ylabel('Proportion of Best Option Selected')
plt.xticks(np.arange(0, 20, 2))
plt.vlines(x=10, ymin=0, ymax=1, color='red', linestyle='--', label='Task Switch')
plt.legend(title='Condition', loc='upper left')
sns.despine()
plt.savefig('./figures/BestOptionByBlock.png', dpi=600)
plt.show()

# Create another figure by condition only
task_2nd_summary = task_2nd.groupby(['Subnum'], observed=False).agg({
    'Condition': 'first',
    'Task': 'first',
    'BestOption': 'mean',
    'HighFreqOption': 'mean'
}).reset_index()


plt.figure(figsize=(10, 6))
sns.barplot(data=task_2nd_summary, x='Condition', y='BestOption', hue='Condition', errorbar='se', palette=sns.color_palette())
plt.xlabel('')
plt.ylabel('Proportion of Best Option Selected')
sns.despine()
plt.savefig('./figures/BestOptionByCondition.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=task_2nd_summary, x='Condition', y='HighFreqOption', hue='Condition', errorbar='se', palette=sns.color_palette())
plt.xlabel('')
plt.ylabel('Proportion of High Frequency Option Selected')
sns.despine()
plt.savefig('./figures/HighFreqOptionByCondition.png', dpi=600)
plt.show()

# Best option
g = sns.catplot(data=task_2nd_summary, x='Condition', y='BestOption', hue='Condition', col='Task', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/BestOptionByCondition_Task.png', dpi=600)
plt.show()

# High frequency option
g = sns.catplot(data=task_2nd_summary, x='Condition', y='HighFreqOption', hue='Condition', col='Task', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of High Frequency Option Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/HighFreqOptionByCondition_Task.png', dpi=600)
plt.show()

# Plot deck selections
g = sns.catplot(data=deck_counts, x='Condition', y='proportion', hue='keyResponse', col='Task', errorbar='ci', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/Deck_Selection_byTask.png', dpi=600)
plt.show()

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

# Calculate deck C and D selections
deck_cd_data = dm_data[dm_data['Task'] == 'IGT'].groupby(['Subnum', 'Condition'], observed=False).agg({
    'keyResponse': lambda x: [(x == 3).mean(), (x == 4).mean()]
}).reset_index()
deck_cd_data = deck_cd_data.dropna()
deck_cd_data[['DeckC', 'DeckD']] = pd.DataFrame(deck_cd_data['keyResponse'].tolist(), index=deck_cd_data.index)
deck_cd_data = deck_cd_data.drop('keyResponse', axis=1)

deck_ab_data = dm_data[dm_data['Task'] == 'IGT'].groupby(['Subnum', 'Condition'], observed=False).agg({
    'keyResponse': lambda x: [(x == 1).mean(), (x == 2).mean()]
}).reset_index()
deck_ab_data = deck_ab_data.dropna()
deck_ab_data[['DeckA', 'DeckB']] = pd.DataFrame(deck_ab_data['keyResponse'].tolist(), index=deck_ab_data.index)
deck_ab_data = deck_ab_data.drop('keyResponse', axis=1)

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
sns.lineplot(data=dm_data, x='Block', y='BestOption', hue='Condition', errorbar='ci')
plt.xlabel('Block Number')
plt.ylabel('Proportion of Best Option Selected')
plt.xticks(np.arange(0, 20, 2))
plt.vlines(x=10, ymin=0, ymax=1, color='red', linestyle='--', label='Task Switch')
plt.legend(title='Condition', loc='upper left')
sns.despine()
plt.savefig('./figures/BestOptionByBlock.png', dpi=600)
plt.show()

# Create another figure by condition only
igt_data = dm_data[dm_data['Task'] == 'IGT']
igt_summary = igt_data.groupby(['Subnum'], observed=False).agg({
    'Condition': 'first',
    'BestOption': 'mean',
    'HighFreqOption': 'mean'
}).reset_index()

#

plt.figure(figsize=(10, 6))
sns.barplot(data=igt_summary, x='Condition', y='BestOption', hue='Condition', errorbar='se', palette=sns.color_palette())
plt.xlabel('')
plt.ylabel('Proportion of Best Option Selected')
sns.despine()
plt.savefig('./figures/BestOptionByCondition.png', dpi=600)
plt.show()

# High frequency option
plt.figure(figsize=(10, 6))
sns.lineplot(data=dm_data, x='Block', y='HighFreqOption', hue='Condition', errorbar='ci')
plt.title('Best Option Selection by Block and Condition')
plt.xlabel('Block Number')
plt.ylabel('Proportion of Best Option Selected')
plt.xticks(np.arange(0, 20, 2))
plt.savefig('./figures/HighFreqOptionByBlock.png', dpi=300)
plt.show()

# Plot deck C and D selections
plt.figure(figsize=(10, 6))
deck_cd_melted = pd.melt(deck_cd_data, id_vars=['Subnum', 'Condition'],
                         value_vars=['DeckC', 'DeckD'],
                         var_name='Deck', value_name='Proportion')
sns.barplot(data=deck_cd_melted, x='Condition', y='Proportion', hue='Deck', errorbar='se')
plt.xlabel('Condition')
plt.ylabel('Proportion Selected')
plt.title('Deck C and D Selection by Condition')
sns.despine()
plt.savefig('./figures/DeckCD_Selection.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 6))
deck_ab_melted = pd.melt(deck_ab_data, id_vars=['Subnum', 'Condition'],
                         value_vars=['DeckA', 'DeckB'],
                         var_name='Deck', value_name='Proportion')
sns.barplot(data=deck_ab_melted, x='Condition', y='Proportion', hue='Deck', errorbar='se')
plt.xlabel('Condition')
plt.ylabel('Proportion Selected')
plt.title('Deck A and B Selection by Condition')
sns.despine()
plt.savefig('./figures/DeckAB_Selection.png', dpi=600)
plt.show()


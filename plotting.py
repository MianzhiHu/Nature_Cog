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
IGT_SGT = dm_data[dm_data['Order'] == 'SGT_IGT'].copy()

# Figure 1: overall performance
IGT_SGT_summary = IGT_SGT.groupby(['Subnum', 'Condition', 'Task', 'Order', 'TaskCode'], observed=False).agg({
    'BestOption': 'mean',
    'HighFreqOption': 'mean'
}).dropna().reset_index()
IGT_SGT_summary['BestOption_z'] = IGT_SGT_summary.groupby(['Task'])['BestOption'].transform(lambda x: (x - x.mean()) / x.std())
IGT_SGT_summary_wide = IGT_SGT_summary.pivot_table(index=['Subnum', 'Condition', 'Order'], columns=['Task'], values='BestOption_z').reset_index()
IGT_SGT_summary_wide['Diff_BestOption_z'] = IGT_SGT_summary_wide['SGT'] - IGT_SGT_summary_wide['IGT']

plt.figure()
sns.barplot(data=IGT_SGT_summary_wide, x='Condition', y='Diff_BestOption_z')
plt.title('Overall Best Option Rates in IGT-SGT')
plt.ylabel('Best Option Rate (SGT - IGT)')
plt.xlabel('Condition')
# plt.ylim(0, 1)
plt.savefig('./figures/Overall_BestOption_Rates_IGT_SGT.png', dpi=600)
plt.close()

# two panel figure for raw performance
plt.figure()
sns.catplot(data=IGT_SGT_summary, x='Task', y='BestOption_z', hue='Condition', kind='bar', errorbar=('se'))
plt.title('Overall Best Option Rates in IGT-SGT')
plt.ylabel('Best Option Rate')
plt.xlabel('Task')
# plt.ylim(0, 1)
plt.savefig('./figures/Overall_BestOption_Rates_IGT_SGT_Raw.png', dpi=600)
plt.close()

# Figure 2: Bestoption rates across conditions arcoss blocks in IGT-SGT
IGT_SGT_summary_blocked = IGT_SGT.groupby(['Subnum', 'Condition', 'Task', 'Order', 'Block', 'TaskCode'], observed=False).agg({
    'BestOption': 'mean',
    'HighFreqOption': 'mean'
}).dropna().reset_index()
IGT_SGT_summary_blocked['BestOption_z'] = IGT_SGT_summary_blocked.groupby(['Task', 'Block'])['BestOption'].transform(lambda x: (x - x.mean()) / x.std())
IGT_SGT_summary_blocked['Block'] = IGT_SGT_summary_blocked['Block'].apply(lambda x: x - 10 if x > 10 else x)
IGT_SGT_summary_blocked_wide = IGT_SGT_summary_blocked.pivot_table(index=['Subnum', 'Condition', 'Order', 'Block'], columns=['Task'], values='BestOption_z').reset_index()
IGT_SGT_summary_blocked_wide['Diff_BestOption_z'] = IGT_SGT_summary_blocked_wide['SGT'] - IGT_SGT_summary_blocked_wide['IGT']

plt.figure()
sns.lineplot(data=IGT_SGT_summary_blocked_wide, x='Block', y='Diff_BestOption_z', hue='Condition', marker='o')
plt.title('Best Option Rates Across Blocks in IGT-SGT')
plt.ylabel('Best Option Rate')
plt.xlabel('Block')
# plt.ylim(0, 1)
plt.legend(title='Task')
plt.savefig('./figures/BestOption_Rates_IGT_SGT.png', dpi=600)
plt.close()

# Figure 3: per deck, calcuate the percentage of each choice
IGT_SGT_deck_summary_deck = (IGT_SGT.groupby(['Subnum', 'Condition', 'Task', 'Order', 'TaskCode'], observed=False)['keyResponse'].
                             value_counts(normalize=True).rename('ChoiceRate').reset_index())
plt.figure()
sns.catplot(data=IGT_SGT_deck_summary_deck, x='keyResponse', y='ChoiceRate', hue='Condition', col='Task', kind='bar', errorbar=('se'))
plt.suptitle('Choice Rates per Deck in IGT-SGT', y=1.05)
plt.ylabel('Choice Rate')
plt.xlabel('Deck (0-3)')
plt.savefig('./figures/Choice_Rates_per_Deck_IGT_SGT.png', dpi=600)
plt.close()
import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import functools
from matplotlib import font_manager as fm
import scipy.stats as stats

# ======================================================================================================================
# Load the data
# ======================================================================================================================
dm_data = pd.read_csv('./data/dm_data.csv')
img_data = pd.read_csv('./data/img_data.csv')
stimuli_info = pd.read_csv('./stimuli/visual_features_with_naturalness.csv')
avg_rating = pd.read_csv('./data/avg_rating.csv')
IGT_SGT = dm_data[dm_data['Order'] == 'IGT_SGT'].copy()

dm_data['Condition'] = pd.Categorical(dm_data['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
task_1st = dm_data[dm_data['TaskCode'] == 1]
task_2nd = dm_data[dm_data['TaskCode'] == 2]
print(f'The number of participants: {dm_data.groupby('Order')['Subnum'].nunique().to_dict()}')
img_count = img_data['image_name'].value_counts().reset_index()
# # extract those who saw MDS
# MDS_140_participants = img_data[img_data['image_name'].str.contains('MDS')]['Subnum'].unique().tolist()

dm_data_sex = dm_data.groupby('Subnum')['Age'].first().reset_index()
dm_data_sex['Age'] = pd.to_numeric(dm_data_sex['Age'], errors='coerce')
print(dm_data_sex['Age'].std())

def z_score(x):
    return (x - x.mean()) / x.std()

# ======================================================================================================================
# Overall Summary
# ======================================================================================================================
dm_summary = dm_data.groupby(['Subnum', 'Condition', 'Task', 'Order', 'TaskCode'], observed=False).agg({
    'BestOption': 'mean',
    'HighFreqOption': 'mean',
    'HighMagOption': 'mean'
}).dropna().reset_index()
dm_summary['Condition'] = pd.Categorical(dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
dm_summary = dm_summary.merge(avg_rating, on=['Subnum'], how='left')

# Calculate optimality scores as (good choices - bad choices)
for metric in ['BestOption', 'HighFreqOption', 'HighMagOption']:
    dm_summary[f'{metric}_Optim'] = dm_summary[metric] - (1 - dm_summary[metric])

# Calculate z-scores grouped by task
for col in ['BestOption', 'HighFreqOption', 'HighMagOption', 'BestOption_Optim', 'HighFreqOption_Optim', 'HighMagOption_Optim']:
    dm_summary[f'{col}_z'] = dm_summary.groupby('Task')[col].transform(z_score)
    dm_summary[f'{col}_z'] = dm_summary.groupby('Task')[col].transform(z_score)

# From dm_data, extract the switch rate and win-stay lose-shift rates
dm_prev = dm_data.copy()
dm_prev['PrevChoice'] = dm_prev.groupby(['Subnum', 'Condition', 'Task', 'Order'])['keyResponse'].shift(1)
dm_prev['PrevOutcome'] = dm_prev.groupby(['Subnum', 'Condition', 'Task', 'Order'])['Reward'].shift(1)
dm_prev['Switch'] = np.where(dm_prev['keyResponse'] != dm_prev['PrevChoice'], 1, 0)
dm_prev['WinStay'] = np.where((dm_prev['PrevOutcome'] > 0) & (dm_prev['keyResponse'] == dm_prev['PrevChoice']), 1, 0)
dm_prev['LoseShift'] = np.where((dm_prev['PrevOutcome'] <= 0) & (dm_prev['keyResponse'] != dm_prev['PrevChoice']), 1, 0)
dm_prev['WSLS'] = dm_prev['WinStay'] + dm_prev['LoseShift']

dm_switch_summary = (dm_prev.groupby(['Subnum', 'Condition', 'Task', 'Order'], observed=True).agg({
    'Switch': 'mean',
    'WinStay': 'mean',
    'LoseShift': 'mean',
    'WSLS': 'mean'
}).reset_index())
dm_switch_summary_wide = dm_switch_summary.pivot_table(index=['Subnum', 'Condition', 'Order'], columns='Task',
                                                      values=['Switch', 'WinStay', 'LoseShift', 'WSLS'], observed=True)
dm_switch_summary_wide.columns = ['_'.join(col).strip() for col in dm_switch_summary_wide.columns.values]
dm_switch_summary_wide = dm_switch_summary_wide.reset_index()
for metric in ['Switch', 'WinStay', 'LoseShift', 'WSLS']:
    diff_col = f'{metric}_Diff'
    dm_switch_summary_wide[diff_col] = np.where(
        dm_switch_summary_wide['Order'] == 'IGT_SGT',
        dm_switch_summary_wide[f'{metric}_SGT'] - dm_switch_summary_wide[f'{metric}_IGT'],
        dm_switch_summary_wide[f'{metric}_IGT'] - dm_switch_summary_wide[f'{metric}_SGT']
    )
    # remove the mean difference
    dm_switch_summary_wide[f'{diff_col}_z'] = dm_switch_summary_wide[diff_col].transform(z_score)

# pivot to wide format
all_metrics = ['BestOption', 'HighFreqOption', 'HighMagOption', 'BestOption_Optim', 'HighFreqOption_Optim',
               'HighMagOption_Optim', 'BestOption_z', 'HighFreqOption_z', 'HighMagOption_z', 'BestOption_Optim_z',
               'HighFreqOption_Optim_z', 'HighMagOption_Optim_z']
dm_summary_task_wide = dm_summary.pivot_table(index=['Subnum', 'Condition', 'Order'], columns='Task', values=all_metrics)
dm_summary_task_wide.columns = ['_'.join(col).strip() for col in dm_summary_task_wide.columns.values]
dm_summary_task_wide = dm_summary_task_wide.reset_index()

# calculate the z score of the difference between tasks
# if order is IGT_SGT, then SGT - IGT; if order is SGT_IGT, then IGT - SGT
for metric in ['BestOption_z', 'HighFreqOption_z', 'HighMagOption_z', 'BestOption_Optim_z', 'HighFreqOption_Optim_z', 'HighMagOption_Optim_z']:
    diff_col = f'{metric}_Diff'
    dm_summary_task_wide[diff_col] = np.where(
        dm_summary_task_wide['Order'] == 'IGT_SGT',
        dm_summary_task_wide[f'{metric}_SGT'] - dm_summary_task_wide[f'{metric}_IGT'],
        dm_summary_task_wide[f'{metric}_IGT'] - dm_summary_task_wide[f'{metric}_SGT']
    )
    # remove the mean difference
    dm_summary_task_wide[f'{diff_col}_z'] = dm_summary_task_wide[diff_col].transform(z_score)

dm_summary_task_wide = pd.merge(dm_summary_task_wide, dm_switch_summary_wide, on=['Subnum', 'Condition', 'Order'], how='left')
dm_summary_task_wide.to_csv('./data/dm_summary_task_wide.csv', index=False)

# now calculate deck selection proportions
deck_summary = (dm_data.groupby(['Subnum', 'Condition', 'Task', 'Order', 'TaskCode'], observed=True)['keyResponse'].
                             value_counts(normalize=True).rename('ChoiceRate').reset_index())
# rename the deck names to Deck_A, Deck_B, etc.
deck_summary['keyResponse'] = deck_summary['keyResponse'].replace({1: 'Deck_A', 2: 'Deck_B', 3: 'Deck_C', 4: 'Deck_D'})
deck_summary['keyResponse'] = pd.Categorical(deck_summary['keyResponse'], categories=['Deck_A', 'Deck_B', 'Deck_C', 'Deck_D'], ordered=True)

# calculate z-scores for choice rates
deck_summary['ChoiceRate_z'] = deck_summary.groupby(['Task', 'keyResponse'], observed=False)['ChoiceRate'].transform(z_score)
deck_summary.to_csv('./data/deck_summary.csv', index=False)
deck_summary_task_wide = deck_summary.pivot_table(index=['Subnum', 'Condition', 'Order', 'Task'], columns=['keyResponse'],
                                                  values=['ChoiceRate', 'ChoiceRate_z'], observed=True)
deck_summary_task_wide.columns = [f'{task}_{deck}' for task, deck in deck_summary_task_wide.columns.to_flat_index()]
dm_summary = pd.merge(dm_summary, deck_summary_task_wide, on=['Subnum', 'Condition', 'Task', 'Order'], how='left')
dm_summary = pd.merge(dm_summary, dm_switch_summary, on=['Subnum', 'Condition', 'Task', 'Order'], how='left')
dm_summary.to_csv('./data/dm_summary.csv', index=False)

task_2nd_summary = dm_summary[dm_summary['TaskCode'] == 2].copy()

# merge these two wide dataframes into dm_data
dm_data_sum = functools.reduce(lambda left, right: pd.merge(left, right, on=['Subnum', 'Condition', 'Order'], how='left'),
                            [dm_data, dm_summary_task_wide, deck_summary_task_wide])
dm_data_sum.to_csv('./data/dm_data_summary.csv', index=False)

# ======================================================================================================================
# IGT-SGT Analysis
# ======================================================================================================================
# IGT_SGT_summary = dm_summary[dm_summary['Order'] == 'IGT_SGT'].copy()
# IGT_SGT_summary = IGT_SGT_summary[IGT_SGT_summary['Task'] == 'SGT']
# IGT_SGT_summary_baseline = dm_summary[(dm_summary['Order'] == 'SGT_IGT') & (dm_summary['Task'] == 'SGT')].copy()
# IGT_SGT_summary_baseline['Condition'] = 'Baseline'
# IGT_SGT_summary = pd.concat([IGT_SGT_summary, IGT_SGT_summary_baseline], ignore_index=True)
IGT_SGT_summary = dm_data[dm_data['Order'] == 'SGT_IGT'].copy()
IGT_SGT_summary = IGT_SGT_summary[IGT_SGT_summary['Task'] == 'IGT']
IGT_SGT_summary_baseline = dm_data[(dm_data['Order'] == 'IGT_SGT') & (dm_data['Task'] == 'IGT')].copy()
IGT_SGT_summary_baseline['Condition'] = 'Baseline'
IGT_SGT_summary = pd.concat([IGT_SGT_summary, IGT_SGT_summary_baseline], ignore_index=True)
IGT_SGT_summary.to_csv('./data/IGT_SGT_summary.csv', index=False)
IGT_SGT_summary_wide = dm_summary_task_wide[dm_summary_task_wide['Order'] == 'IGT_SGT'].copy()
IGT_SGT_summary_wide = pd.merge(IGT_SGT_summary_wide, avg_rating[['Subnum', 'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']], on='Subnum', how='left')
dm_summary_modeled = pd.read_csv('./data/dm_summary_modeled.csv')
dm_summary_modeled_wide = pd.read_csv('./data/dm_summary_modeled_wide.csv')
dm_summary_modeled_wide['Condition'] = pd.Categorical(dm_summary_modeled_wide['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
print(IGT_SGT_summary_wide.shape)

# one sample t-test on the difference scores
p = []
for con in ['Nature', 'Urban', 'Control']:
    subset = IGT_SGT_summary_wide[IGT_SGT_summary_wide['Condition'] == con]
    ttest_result = pg.ttest(subset['BestOption_z_Diff'], 0)
    print(f'One-sample t-test for BestOption_z_Diff in {con} condition:')
    print(f'Mean difference: {subset["BestOption_z_Diff"].mean():.4f}')
    print(f't-statistic: {ttest_result["T"].values[0]:.4f}, p-value: {ttest_result["p-val"].values[0]:.4f}, n={len(subset)}')
    p.append(ttest_result["p-val"].values[0])

anova = pg.anova(dv='BestOption_z_Diff', between='Condition', data=dm_summary_task_wide, detailed=True)
pairwise = pg.pairwise_tests(dv='BestOption_z_Diff', between='Condition', data=IGT_SGT_summary_wide, padjust='fdr_bh')
print(anova)
print(pairwise)

# adjust p-values for multiple comparisons
p_adjusted = pg.multicomp(pvals=p, method='fdr_bh')
print(f'Adjusted p-values for one-sample t-tests: {p_adjusted[1]}')

# Plot IGT-SGT results
# # keep MDS140 participants only for the nature condition
# dm_summary_subset = dm_summary_task_wide[(dm_summary_task_wide['Condition'] == 'Nature') & (dm_summary_task_wide['Subnum'].isin(MDS_140_participants))].copy()
# dm_summary_subset = pd.concat([dm_summary_subset,
#                                dm_summary_task_wide[dm_summary_task_wide['Condition'] != 'Nature']], ignore_index=True)
# dm_summary_subset['Condition'] = pd.Categorical(dm_summary_subset['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
dm_summary['Condition'] = pd.Categorical(dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
g = sns.catplot(data=dm_summary, x='Condition', y='BestOption', hue='Condition', row='Task', col='Order', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
g.set_titles('{col_name} - {row_name}')
g.despine()
plt.savefig('./figures/BestOptionByCondition_IGT_SGT.png', dpi=600)
plt.show()

# difference
palette = sns.color_palette('deep')
nature_color = palette[2]
urban_color = palette[3]
control_color = palette[7]
palette_custom = [nature_color, urban_color, control_color]

# import font
font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)
def upward_only(group):
    mean = group.mean()
    se = group.sem()
    return (np.zeros_like(se), se)   # lower=0, upper=se

g = sns.catplot(data=IGT_SGT_summary_wide, x='Condition', y='BestOption_Optim_z_Diff', hue='Condition', errorbar='se', kind='bar',
                height=5, aspect=1.2, palette=palette_custom)
g.set_axis_labels('', 'Performance Improvement (z-score)', fontproperties=prop)
g.set_xticklabels(fontproperties=prop, fontsize=18)
g.set_yticklabels(fontproperties=prop, fontsize=12)
# x and y labels
for ax in g.axes.flat:
    ax.yaxis.label.set_fontproperties(prop)
    ax.yaxis.label.set_fontsize(20)
g.despine()
plt.title('IGT-SGT', fontproperties=prop, fontsize=22)
plt.tight_layout()
plt.savefig('./figures/within_subj_diff.png', dpi=600)
plt.show()

g = sns.catplot(data=dm_summary_modeled_wide, x='Condition', y='la_Diff_z', hue='Condition', errorbar='se', kind='bar',
                height=5, aspect=1.2, palette=palette_custom)
g.set_axis_labels('', 'Loss Aversion Change (z-score)', fontproperties=prop)
# x and y tick labels
g.set_xticklabels(fontproperties=prop, fontsize=18)
g.set_yticklabels(fontproperties=prop, fontsize=12)
# x and y labels
for ax in g.axes.flat:
    ax.yaxis.label.set_fontproperties(prop)
    ax.yaxis.label.set_fontsize(20)
g.despine()
plt.tight_layout()
g.despine()
plt.savefig('./figures/tByCondition_IGT_SGT.png', dpi=600)
plt.show()

img_rating_summary = img_data.groupby(['image_name', 'Condition']).agg({
    'naturalness': 'mean',
    'disorderliness': 'mean',
    'aesthetic': 'mean',
}).reset_index()
# img_rating_summary = img_rating_summary[img_rating_summary['Condition'] != 'Control']

# run correlation analysis
print(pg.corr(img_rating_summary['naturalness'], img_rating_summary['disorderliness'], method='pearson'))
print(pg.corr(img_rating_summary['naturalness'], img_rating_summary['aesthetic'], method='pearson'))
print(pg.corr(img_rating_summary['disorderliness'], img_rating_summary['aesthetic'], method='pearson'))

# mixed effects model
me_model = ols('disorderliness ~ aesthetic + C(Condition) + (1|Subnum)', data=img_data).fit()
print(me_model.summary())

# Process image data
# presence matrix for each image in each participant (1 if present, 0 if not) with each column as image_name
img_data = img_data[img_data['Order'] == 'IGT_SGT'].copy()
img_presence = img_data.pivot_table(index="Subnum", columns="image_name", values="Condition", aggfunc="count", fill_value=0)
img_presence  = (img_presence > 0).astype(int)

performance = IGT_SGT_summary_wide['BestOption_Optim_z_Diff']
influences = {}

for img in img_presence.columns:
    influences[img] = stats.spearmanr(img_presence[img], performance)[0]
    print(stats.spearmanr(img_presence[img], performance))

influence_df = pd.DataFrame({
    "ImageName": list(influences.keys()),
    "Influence": list(influences.values()),
    "Condition": [img_rating_summary[img_rating_summary['image_name'] == img]['Condition'].values[0] for img in influences.keys()]
})
t_test = pg.ttest(influence_df[influence_df['Condition'] == 'Control']['Influence'], 0)
print(influence_df.groupby('Condition')['Influence'].mean())

stimuli_info = stimuli_info.merge(influence_df, on='ImageName', how='left')
stimuli_info = stimuli_info.merge(img_rating_summary[['image_name', 'naturalness', 'disorderliness', 'aesthetic']], left_on='ImageName', right_on='image_name', how='left')
stimuli_info.to_csv('./stimuli/stimuli_influence.csv', index=False)

plt.figure(figsize=(8,6))
sns.regplot(data=stimuli_info, x="Semantic_PC1", y="Influence", scatter=False, order=1)
sns.scatterplot(data=stimuli_info, x="Semantic_PC1", y="Influence", hue="Condition", palette=palette_custom, s=100, edgecolor='black')

plt.xlabel("Naturalness (PCA)")
plt.ylabel("Image Influence on Behavior (z-diff)")
plt.title("Stimulus-level Effects on Behavioral Performance")
plt.savefig('./figures/Semantic_PC11-influence.png', dpi=600)
plt.show()
# ======================================================================================================================
# Plotting
# ======================================================================================================================
# Create a correlation plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='naturalness', y='Perc_Nat', hue='Condition', data=img_rating_summary, alpha=0.5)
# sns.regplot(x='naturalness', y='Perc_Nat', data=img_rating_summary, scatter=False,
#             line_kws={'color': 'red', 'linewidth': 2})
# plt.xlabel('Observed Naturalness Rating')
# plt.ylabel('Original Naturalness Rating')
# plt.legend(title='Condition')
# sns.despine()
# plt.tight_layout()
# plt.savefig('./figures/Naturalness_vs_Original.png', dpi=600)
# plt.show()

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

# Best option
g = sns.catplot(data=task_2nd_summary, x='Condition', y='BestOption', hue='Condition', col='Task', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/BestOptionByCondition_Task_2nd.png', dpi=600)
plt.show()

# difference
g = sns.catplot(data=dm_summary_task_wide, x='Condition', y='BestOption_Optim_z_Diff', hue='Condition', col='Order', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/within_subj_diff.png', dpi=600)
plt.show()

# g = sns.scatterplot(data=IGT_SGT_summary_wide, x='BestOption_Optim_z_Diff', y='Semantic_PC1', hue='Condition')
# g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
# g.set_titles('{col_name}')
# g.despine()
# plt.savefig('./figures/within_subj_diff.png', dpi=600)
# plt.show()

# High frequency option
g = sns.catplot(data=dm_summary, x='Condition', y='BestOption', hue='Condition', row='Task', col='Order', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of High Frequency Option Selected')
g.set_titles('{col_name}')
g.despine()
plt.savefig('./figures/HighFreqOptionByCondition_Task.png', dpi=600)
plt.show()

# Plot deck selections
g = sns.catplot(data=deck_summary, x='Condition', y='ChoiceRate_z', hue='keyResponse', row='Task', col='Order',
                errorbar='ci', kind='bar', height=5, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion Selected')
for ax, row_val in zip(g.axes[:,0], g.row_names):
    ax.set_ylabel(f'Task: {row_val}', fontsize=20)
for ax, col_val in zip(g.axes[0], g.col_names):
    ax.set_title(f'Order: {col_val}', fontsize=20)
g.despine()
# plt.tight_layout()
plt.savefig('./figures/Deck_Selection_byTask.png', dpi=600)
plt.show()

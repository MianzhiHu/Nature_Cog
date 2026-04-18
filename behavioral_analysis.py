from cmath import nan

import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import functools
from matplotlib import font_manager as fm
import scipy.stats as stats
from utils.ComputationalModeling import behavioral_moving_window

# ======================================================================================================================
# Load the data
# ======================================================================================================================
E1_dm_data = pd.read_csv('./data/E1_dm_data.csv')
E1_dm_modeled = pd.read_csv('./data/dm_summary_modeled.csv')
E1_exploration = pd.read_csv('./data/exploration_data.csv')

E1_img_data = pd.read_csv('./data/E1_img_data.csv')
E1_avg_rating = pd.read_csv('./data/E1_avg_rating.csv')
E2_data = pd.read_csv('./data/E2_all_data.csv')
E2_avg_rating = pd.read_csv('./data/E2_avg_rating.csv')
stimuli_info = pd.read_csv('./stimuli/visual_features_with_naturalness.csv')

E1_exploration['Trial'] = E1_exploration.groupby(['Subnum', 'Condition', 'Task']).cumcount() + 2
E1_dm_data['Condition'] = pd.Categorical(E1_dm_data['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
E2_data['Condition'] = pd.Categorical(E2_data['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
task_1st = E1_dm_data[E1_dm_data['Task'] == 1]
task_2nd = E1_dm_data[E1_dm_data['Task'] == 2]
print(f'The number of participants: {E1_dm_data.groupby('Condition', observed=False)['Subnum'].nunique().to_dict()}')
img_count = E1_img_data['image_name'].value_counts().reset_index()

font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)
palette = sns.color_palette('deep')
nature_color = palette[2]
urban_color = palette[3]
control_color = palette[7]
palette_custom = [nature_color, urban_color, control_color]
# 

# Demographic information
E1_dm_data_sex = E1_dm_data.groupby('Subnum')['Age'].first().reset_index()
E1_dm_data_sex['Age'] = pd.to_numeric(E1_dm_data_sex['Age'], errors='coerce')
print(E1_dm_data_sex['Age'].std())

def z_score(x):
    return (x - x.mean()) / x.std()

# ======================================================================================================================
# E1 Analysis
# ======================================================================================================================
E1_dm_summary = E1_dm_data.groupby(['Subnum', 'Condition', 'Task'], observed=False).agg({
    'BestChoice': 'mean',
    'Reward': 'mean'
}).dropna().reset_index()
E1_dm_summary['BestChoice_z'] = E1_dm_summary.groupby('Task')['BestChoice'].transform(z_score)
E1_dm_summary['Condition'] = pd.Categorical(E1_dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
E1_dm_summary = E1_dm_summary.merge(E1_avg_rating, on=['Subnum'], how='left')

# ----------------------------------------------------------------------------------------------------------------------
# WSLS analysis
# ----------------------------------------------------------------------------------------------------------------------
# From E1_dm_data, extract the switch rate and win-stay lose-shift rates
dm_prev = E1_dm_data.copy()
dm_prev = pd.merge(dm_prev, E1_exploration, on=['Subnum', 'Condition', 'Task', 'Trial'], how='left')
dm_prev['exploration'] = dm_prev['exploration'].map({'exploitation': 0, 'exploration': 1})
dm_prev['Average'] = (dm_prev.groupby(['Subnum', 'Condition', 'Task'])['Reward'].expanding().mean().reset_index(level=[0,1,2], drop=True))
dm_prev['PrevChoice'] = dm_prev.groupby(['Subnum', 'Condition', 'Task'])['KeyResponse'].shift(1)
dm_prev['PrevOutcome'] = dm_prev.groupby(['Subnum', 'Condition', 'Task'])['Reward'].shift(1)
dm_prev['PrevOutcome2'] = dm_prev.groupby(['Subnum', 'Condition', 'Task'])['Reward'].shift(2)
dm_prev['PrevAverage2'] = dm_prev.groupby(['Subnum', 'Condition', 'Task'])['Average'].shift(2)
dm_prev['Switch'] = np.where(dm_prev['KeyResponse'] != dm_prev['PrevChoice'], 1, 0)
dm_prev['WinStay'] = np.where((dm_prev['PrevOutcome'] > dm_prev['PrevAverage2']) & (dm_prev['KeyResponse'] == dm_prev['PrevChoice']), 1, 0)
dm_prev['LoseShift'] = np.where((dm_prev['PrevOutcome'] <= dm_prev['PrevAverage2']) & (dm_prev['KeyResponse'] != dm_prev['PrevChoice']), 1, 0)
dm_prev['WSLS'] = dm_prev['WinStay'] + dm_prev['LoseShift']
# dm_prev['grass_weighted'] = dm_prev['grass'] * 0.5487266429826448
# dm_prev['river_weighted'] = dm_prev['river'] * 0.39241939536531545
# dm_prev['water_weighted'] = dm_prev['water'] * 0.34653498068717214
# dm_prev['sidewalk_weighted'] = dm_prev['sidewalk'] * 0.3452173757477936
# dm_prev['bench_weighted'] = dm_prev['bench'] * 0.2907458612546477
# dm_prev['Semantic_agg'] = dm_prev[['grass_weighted', 'river_weighted', 'water_weighted', 'sidewalk_weighted', 'bench_weighted']].sum(axis=1)
# dm_prev = dm_prev[dm_prev['Condition'] != 'Control'].copy()
# dm_prev['Semantic_agg'] = dm_prev['Semantic_agg'].transform(z_score)
# dm_prev.to_csv('./data/dm_switch.csv', index=False)


# dm_prev.loc[dm_prev['Semantic_agg'] > 1, 'agg_Condition'] = 'High Composite Score'
# dm_prev.loc[dm_prev['Semantic_agg'] < -1, 'agg_Condition'] = 'Low Composite Score'
# dm_prev.loc[(dm_prev['Semantic_agg'] <= 1) & (dm_prev['Semantic_agg'] >= -1), 'agg_Condition'] = 'Mid Composite Score'
# dm_prev_only = dm_prev[dm_prev['agg_Condition'].notna()].copy()
# dm_prev_only.to_csv('./data/agg_condition_value_counts.csv', index=False)

# # Testing!
# # Moving window switch rate analysis (30 trials)
# results = {
#     'BestChoice':[],
#     'Reward': [],
#     'Switch': [],
#     'WinStay': [],
#     'LoseShift': [],
#     'WSLS': [],
#     'exploration':[]
# }
#
# for (subnum, condition, task), group in dm_prev.groupby(['Subnum', 'Condition', 'Task'], observed=True):
#     for metric in results.keys():
#         result_df = behavioral_moving_window(group, metric)
#         result_df['Subnum'] = subnum
#         result_df['Condition'] = condition
#         result_df['Task'] = task
#         results[metric].append(result_df)
#
# # Concatenate all results
# mw_best_df = pd.concat(results['BestChoice'], ignore_index=True)
# mw_reward_df = pd.concat(results['Reward'], ignore_index=True)
# mw_switch_df = pd.concat(results['Switch'], ignore_index=True)
# mw_ws_df = pd.concat(results['WinStay'], ignore_index=True)
# mw_ls_df = pd.concat(results['LoseShift'], ignore_index=True)
# mw_wsls_df = pd.concat(results['WSLS'], ignore_index=True)
# mw_exploration_df = pd.concat(results['exploration'], ignore_index=True)
#
# # Combine all metrics into a single DataFrame
# moving_window_df = mw_switch_df.copy()
# moving_window_df['BestChoice'] = mw_best_df['BestChoice']
# moving_window_df['Reward'] = mw_reward_df['Reward']
# moving_window_df['WinStay'] = mw_ws_df['WinStay']
# moving_window_df['LoseShift'] = mw_ls_df['LoseShift']
# moving_window_df['WSLS'] = mw_wsls_df['WSLS']
# moving_window_df['Exploration'] = mw_exploration_df['exploration']
# moving_window_df.to_csv('./data/E1_behavioral_moving_window.csv', index=False)
#
# # Plot moving window switch rate trajectory
# plt.figure(figsize=(12, 8))
# for task in [1, 2]:
#     plt.subplot(2, 1, task)
#     task_data = moving_window_df[moving_window_df['Task'] == task]
#     sns.lineplot(data=task_data, x='Trial', y='Switch', hue='Condition', errorbar=('se'),
#                  ax=plt.gca())
#     plt.xlabel('Trial (Window Start)')
#     plt.ylabel('Switch Rate (30-trial window)')
#     plt.title(f'Task {task}: Moving Window Switch Rate Trajectory')
#     plt.legend(title='Condition')
#     plt.grid(True, alpha=0.3)
# plt.tight_layout()
# sns.despine()
# plt.savefig('./figures/moving_switch_rate_trajectory.png', dpi=600)
# plt.show()



dm_switch_summary = (dm_prev.groupby(['Subnum', 'Condition', 'agg_Condition', 'Task'], observed=True).agg({
    'BestChoice': 'mean',
    'Switch': 'mean',
    'WinStay': 'mean',
    'LoseShift': 'mean',
    'WSLS': 'mean'
}).reset_index())
value_counts_df = (dm_switch_summary.groupby('Condition')['agg_Condition'].value_counts() / 2).reset_index(name='count')
print(value_counts_df)


# # Plot the value counts
# plt.figure(figsize=(10, 6))
# sns.barplot(data=value_counts_df, x='Condition', y='count', hue='agg_Condition', palette=palette_custom[:2])
# plt.title('Composite Score +/-1 SD per Condition', fontproperties=prop, fontsize=20)
# plt.xlabel('')
# plt.ylabel('Frequency', fontproperties=prop, fontsize=16)
# ax = plt.gca()
# for lbl in ax.get_xticklabels():
#     lbl.set_fontproperties(prop)
#     lbl.set_fontsize(14)
# for lbl in ax.get_yticklabels():
#     lbl.set_fontproperties(prop)
#     lbl.set_fontsize(14)
# legend = ax.get_legend()
# if legend is not None:
#     legend.set_title('Composite Score Category')
#     plt.setp(legend.get_title(), fontproperties=prop, fontsize=16)
#     plt.setp(legend.get_texts(), fontproperties=prop, fontsize=14)
# sns.despine()
# plt.tight_layout()
# plt.savefig('./figures/agg_condition_value_counts.png', dpi=600)
# plt.show()

# dm_switch_summary.to_csv('./data/dm_switch_summary.csv', index=False)

print(f'Switch Rate: {dm_switch_summary.groupby(['Condition', 'Task'])["Switch"].mean()}')
print('=' * 50)
print(f'Win-Stay: {dm_switch_summary.groupby(['Condition', 'Task'])['WinStay'].mean()}')
print('=' * 50)
print(f'Lose-Shift: {dm_switch_summary.groupby(['Condition', 'Task'])["LoseShift"].mean()}')
print('=' * 50)
print(f'WSLS: {dm_switch_summary.groupby(['Condition', 'Task'])["WSLS"].mean()}')

dm_switch_summary_wide = dm_switch_summary.pivot_table(index=['Subnum', 'Condition'], columns='Task',
                                                      values=['BestChoice', 'Switch', 'WinStay', 'LoseShift', 'WSLS'], observed=True)
dm_switch_summary_wide.columns = ['_'.join(map(str, col)).strip()for col in dm_switch_summary_wide.columns.values]

dm_switch_summary_wide = dm_switch_summary_wide.reset_index()
for metric in ['BestChoice', 'Switch', 'WinStay', 'LoseShift', 'WSLS']:
    # z-score each task separately
    dm_switch_summary_wide[f'{metric}_1_z'] = dm_switch_summary_wide[f'{metric}_1'].transform(z_score)
    dm_switch_summary_wide[f'{metric}_2_z'] = dm_switch_summary_wide[f'{metric}_2'].transform(z_score)
    # calculate difference of z-scores
    diff_col = f'{metric}_Diff'
    dm_switch_summary_wide[diff_col] = dm_switch_summary_wide[f'{metric}_2_z'] - dm_switch_summary_wide[f'{metric}_1_z']
    # z-score the difference
    dm_switch_summary_wide[f'{diff_col}_z'] = dm_switch_summary_wide[diff_col].transform(z_score)
    


# Statistically test
wsls_results = []
for metric in ['Switch', 'WinStay', 'LoseShift', 'WSLS']:
    anova = pg.mixed_anova(dv=f'{metric}', between='Condition', within='Task', data=dm_switch_summary, subject='Subnum')
    wsls_results.append((metric, anova))
pairwise = pg.pairwise_tests(dv='Switch', between='Condition', within='Task', data=dm_switch_summary, subject='Subnum', padjust='fdr_bh')

# Plot
plot_df = dm_switch_summary.copy()
plot_df['Condition'] = pd.Categorical(plot_df['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
# plot_df['agg_Condition'] = pd.Categorical(plot_df['agg_Condition'], categories=['High Composite Score', 'Mid Composite Score', 'Low Composite Score'], ordered=True)
plot_df['Task'] = plot_df['Task'].map({1: 'First', 2: 'Second'})
plot_df['Task'] = pd.Categorical(plot_df['Task'], categories=['First', 'Second'], ordered=True)


for metric in ['BestChoice', 'Switch', 'WinStay', 'LoseShift', 'WSLS']:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df, x='Condition', y=metric, hue='Task', errorbar='se', palette=palette_custom)
    plt.title(f'')
    plt.xlabel('')
    plt.ylabel('P(Switch)', fontproperties=prop, fontsize=20)
    ax = plt.gca()
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(16)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(16)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_loc('lower left')
        legend.set_alpha(0.5)
        plt.setp(legend.get_title(), fontproperties=prop, fontsize=18)
        plt.setp(legend.get_texts(), fontproperties=prop, fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'./figures/{metric}_by_Condition_and_Task.png', dpi=600)
    plt.show()
    plt.clf()

# Plot z-scored differences for Switch, WinStay, LoseShift, WSLS
dm_switch_summary_wide['Condition'] = pd.Categorical(dm_switch_summary_wide['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
for metric in ['BestChoice', 'Switch', 'WinStay', 'LoseShift', 'WSLS']:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=dm_switch_summary_wide, x='Condition', y=f'{metric}_Diff', hue='Condition', errorbar='se',
                palette=palette_custom, legend=False, dodge=False)
    plt.title(f'')
    plt.xlabel('')
    plt.ylabel(f'{metric} Change (z-score)', fontproperties=prop, fontsize=20)
    ax = plt.gca()
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(16)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(16)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'./figures/{metric}_Diff_z_by_Condition.png', dpi=600)
    plt.show()
    plt.close()

agg_condition = dm_switch_summary[['Subnum', 'Condition', 'agg_Condition', 'Task']]
E1_dm_modeled = pd.merge(E1_dm_modeled, agg_condition, on=['Subnum', 'Condition', 'Task'], how='left')
E1_dm_modeled = E1_dm_modeled[~E1_dm_modeled['agg_Condition'].isna()].copy()
E1_dm_modeled['agg_Condition'] = pd.Categorical(E1_dm_modeled['agg_Condition'], categories=['High Composite Score', 'Mid Composite Score', 'Low Composite Score'], ordered=True)
E1_dm_modeled['Task'] = E1_dm_modeled['Task'].map({1: 'First', 2: 'Second'})

plt.figure(figsize=(8, 6))
sns.barplot(data=E1_dm_modeled, x='agg_Condition', y='Exploration_Rate', hue='Task', errorbar='se', palette=palette_custom)
plt.title(f'')
plt.xlabel('')
plt.ylabel('Exploration Rate', fontproperties=prop, fontsize=20)
ax = plt.gca()
for lbl in ax.get_xticklabels():
    lbl.set_fontproperties(prop)
    lbl.set_fontsize(16)
for lbl in ax.get_yticklabels():
    lbl.set_fontproperties(prop)
    lbl.set_fontsize(16)
legend = ax.get_legend()
if legend is not None:
    legend.set_title('Task')
    legend.set_loc('lower left')
    legend.set_alpha(0.5)
    plt.setp(legend.get_title(), fontproperties=prop, fontsize=18)
    plt.setp(legend.get_texts(), fontproperties=prop, fontsize=16)
sns.despine()
plt.tight_layout()
plt.savefig(f'./figures/exploration_by_aggCondition_and_Task.png', dpi=600)
plt.show()
plt.clf()

# pivot to wide format
all_metrics = ['BestChoice', 'BestChoice_z']
E1_dm_summary_task_wide = E1_dm_summary.pivot_table(index=['Subnum', 'Condition'], columns='Task', values=all_metrics)
E1_dm_summary_task_wide.columns = ['_'.join(map(str, col)).strip() for col in E1_dm_summary_task_wide.columns.values]
E1_dm_summary_task_wide = E1_dm_summary_task_wide.reset_index()

# calculate the z score of the difference between tasks
# if order is IGT_SGT, then SGT - IGT; if order is SGT_IGT, then IGT - SGT
for metric in ['BestChoice', 'BestChoice_z']:
    diff_col = f'{metric}_Diff'
    E1_dm_summary_task_wide[diff_col] =  E1_dm_summary_task_wide[f'{metric}_2'] - E1_dm_summary_task_wide[f'{metric}_1']
    # remove the mean difference
    E1_dm_summary_task_wide[f'{diff_col}_z'] = E1_dm_summary_task_wide[diff_col].transform(z_score)

E1_dm_summary_task_wide = pd.merge(E1_dm_summary_task_wide, dm_switch_summary_wide, on=['Subnum', 'Condition'], how='left')
E1_dm_summary_task_wide.to_csv('./data/E1_dm_summary_task_wide.csv', index=False)
E1_dm_summary = pd.merge(E1_dm_summary, dm_switch_summary, on=['Subnum', 'Condition', 'Task', 'BestChoice'], how='left')
E1_dm_summary.to_csv('./data/E1_dm_summary.csv', index=False)

# merge these two wide dataframes into E1_dm_data
E1_dm_data_sum = functools.reduce(lambda left, right: pd.merge(left, right, on=['Subnum', 'Condition'], how='left'),
                            [E1_dm_data, E1_dm_summary_task_wide])
E1_dm_data_sum.to_csv('./data/E1_dm_data_summary.csv', index=False)

# ======================================================================================================================
# E2 Analysis
# ======================================================================================================================
E2_dm_summary = E2_data.groupby(['Subnum', 'Condition'], observed=False).agg({
    'BestChoice': 'mean',
    'Reward': 'mean'
}).dropna().reset_index()
E2_dm_summary['BestChoice_z'] = E2_dm_summary['BestChoice'].transform(z_score)
E2_dm_summary['Condition'] = pd.Categorical(E2_dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
E2_dm_summary = E2_dm_summary.merge(E2_avg_rating, on=['Subnum'], how='left')

# ----------------------------------------------------------------------------------------------------------------------
# WSLS analysis
# ----------------------------------------------------------------------------------------------------------------------
# From E2_dm_data, extract the switch rate and win-stay lose-shift rates
dm_prev = E2_data.copy()
dm_prev['Average'] = (dm_prev.groupby(['Subnum', 'Condition'])['Reward'].expanding().mean().reset_index(level=[0,1,2], drop=True))
dm_prev['PrevChoice'] = dm_prev.groupby(['Subnum', 'Condition'])['KeyResponse'].shift(1)
dm_prev['PrevOutcome'] = dm_prev.groupby(['Subnum', 'Condition'])['Reward'].shift(1)
dm_prev['PrevOutcome2'] = dm_prev.groupby(['Subnum', 'Condition'])['Reward'].shift(2)
dm_prev['PrevAverage2'] = dm_prev.groupby(['Subnum', 'Condition'])['Average'].shift(2)
dm_prev['Switch'] = np.where(dm_prev['KeyResponse'] != dm_prev['PrevChoice'], 1, 0)
dm_prev['WinStay'] = np.where((dm_prev['PrevOutcome'] > dm_prev['PrevAverage2']) & (dm_prev['KeyResponse'] == dm_prev['PrevChoice']), 1, 0)
dm_prev['LoseShift'] = np.where((dm_prev['PrevOutcome'] <= dm_prev['PrevAverage2']) & (dm_prev['KeyResponse'] != dm_prev['PrevChoice']), 1, 0)
dm_prev['WSLS'] = dm_prev['WinStay'] + dm_prev['LoseShift']
dm_prev.to_csv('./data/E2_dm_switch.csv', index=False)

dm_switch_summary = (dm_prev.groupby(['Subnum', 'Condition'], observed=True).agg({
    'Switch': 'mean',
    'WinStay': 'mean',
    'LoseShift': 'mean',
    'WSLS': 'mean'
}).reset_index())
dm_switch_summary.to_csv('./data/dm_switch_summary.csv', index=False)

print(f'Switch Rate: {dm_switch_summary.groupby('Condition')["Switch"].mean()}')
print('=' * 50)
print(f'Win-Stay: {dm_switch_summary.groupby('Condition')['WinStay'].mean()}')
print('=' * 50)
print(f'Lose-Shift: {dm_switch_summary.groupby('Condition')["LoseShift"].mean()}')
print('=' * 50)
print(f'WSLS: {dm_switch_summary.groupby('Condition')["WSLS"].mean()}')


# ======================================================================================================================
# IGT-SGT Analysis
# ======================================================================================================================
# IGT_SGT_summary = E2_dm_summary[E1_dm_summary['Order'] == 'IGT_SGT'].copy()
# IGT_SGT_summary = IGT_SGT_summary[IGT_SGT_summary['Task'] == 'SGT']
# IGT_SGT_summary_baseline = E1_dm_summary[(E1_dm_summary['Order'] == 'SGT_IGT') & (E1_dm_summary['Task'] == 'SGT')].copy()
# IGT_SGT_summary_baseline['Condition'] = 'Baseline'
# IGT_SGT_summary = pd.concat([IGT_SGT_summary, IGT_SGT_summary_baseline], ignore_index=True)
IGT_SGT_summary = E1_dm_data[E1_dm_data['Order'] == 'SGT_IGT'].copy()
IGT_SGT_summary = IGT_SGT_summary[IGT_SGT_summary['Task'] == 'IGT']
IGT_SGT_summary_baseline = E1_dm_data[(E1_dm_data['Order'] == 'IGT_SGT') & (E1_dm_data['Task'] == 'IGT')].copy()
IGT_SGT_summary_baseline['Condition'] = 'Baseline'
IGT_SGT_summary = pd.concat([IGT_SGT_summary, IGT_SGT_summary_baseline], ignore_index=True)
IGT_SGT_summary.to_csv('./data/IGT_SGT_summary.csv', index=False)
IGT_SGT_summary_wide = E1_dm_summary_task_wide[E1_dm_summary_task_wide['Order'] == 'IGT_SGT'].copy()
IGT_SGT_summary_wide = pd.merge(IGT_SGT_summary_wide, E1_avg_rating[['Subnum', 'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']], on='Subnum', how='left')
E1_dm_summary_modeled = pd.read_csv('./data/E1_dm_summary_modeled.csv')
E1_dm_summary_modeled_wide = pd.read_csv('./data/E1_dm_summary_modeled_wide.csv')
E1_dm_summary_modeled_wide['Condition'] = pd.Categorical(E1_dm_summary_modeled_wide['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
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

anova = pg.anova(dv='BestOption_z_Diff', between='Condition', data=E1_dm_summary_task_wide, detailed=True)
pairwise = pg.pairwise_tests(dv='BestOption_z_Diff', between='Condition', data=IGT_SGT_summary_wide, padjust='fdr_bh')
print(anova)
print(pairwise)

# adjust p-values for multiple comparisons
p_adjusted = pg.multicomp(pvals=p, method='fdr_bh')
print(f'Adjusted p-values for one-sample t-tests: {p_adjusted[1]}')

# Plot IGT-SGT results
# # keep MDS140 participants only for the nature condition
# E1_dm_summary_subset = E1_dm_summary_task_wide[(E1_dm_summary_task_wide['Condition'] == 'Nature') & (E1_dm_summary_task_wide['Subnum'].isin(MDS_140_participants))].copy()
# E1_dm_summary_subset = pd.concat([E1_dm_summary_subset,
#                                E1_dm_summary_task_wide[E1_dm_summary_task_wide['Condition'] != 'Nature']], ignore_index=True)
# E1_dm_summary_subset['Condition'] = pd.Categorical(E1_dm_summary_subset['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
E1_dm_summary['Condition'] = pd.Categorical(E1_dm_summary['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
g = sns.catplot(data=E1_dm_summary, x='Condition', y='BestOption', hue='Condition', row='Task', col='Order', errorbar='se', kind='bar',
                height=4, aspect=1.2)
g.set_axis_labels('Condition', 'Proportion of Best Option Selected')
g.set_titles('{col_name} - {row_name}')
g.despine()
plt.savefig('./figures/BestOptionByCondition_IGT_SGT.png', dpi=600)
plt.show()

# difference
palette = sns.color_palette('deep')
nature_color = palette[2]
urban_color = palette[5]
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

g = sns.catplot(data=E1_dm_summary_modeled_wide, x='Condition', y='la_Diff_z', hue='Condition', errorbar='se', kind='bar',
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

img_rating_summary = E1_img_data.groupby(['image_name', 'Condition']).agg({
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
me_model = ols('disorderliness ~ aesthetic + C(Condition) + (1|Subnum)', data=E1_img_data).fit()
print(me_model.summary())

# Process image data
# presence matrix for each image in each participant (1 if present, 0 if not) with each column as image_name
E1_img_data = E1_img_data[E1_img_data['Order'] == 'IGT_SGT'].copy()
img_presence = E1_img_data.pivot_table(index="Subnum", columns="image_name", values="Condition", aggfunc="count", fill_value=0)
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
g = sns.catplot(data=E1_dm_summary_task_wide, x='Condition', y='BestOption_Optim_z_Diff', hue='Condition', col='Order', errorbar='se', kind='bar',
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
g = sns.catplot(data=E1_dm_summary, x='Condition', y='BestOption', hue='Condition', row='Task', col='Order', errorbar='se', kind='bar',
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

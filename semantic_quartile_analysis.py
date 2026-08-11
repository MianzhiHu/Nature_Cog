import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from matplotlib import font_manager as fm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from pls_data_parser import semantic_visual_features


result_dir = os.path.abspath(
    'C:/Users/zuire/OneDrive/桌面/胡勉之/Texas A&M University/PLS/Result/Nature_Cog/'
)
lv_path = os.path.join(result_dir, 'PLS_model~semantic_lv_vals.mat')
boot_ratio_path = os.path.join(result_dir, 'PLS_model~semantic.mat')

target_features = ['grass', 'water', 'river']
group_col = 'Weighted_Semantic_Condition'
condition_order = ['Low Weighted Semantic', 'Middle Weighted Semantic', 'High Weighted Semantic']
task_order = ['First', 'Second']

font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
prop = fm.FontProperties(fname=font_path)
palette = sns.color_palette('deep')
low_color = palette[3]
middle_color = palette[7]
high_color = palette[2]
task_palette = {'First': low_color, 'Second': high_color}


def get_model_semantic_weights(lv=1):
    col = lv - 1
    u1 = sio.loadmat(lv_path, variable_names=['u1'])['u1'][:, col]
    boot_ratio = sio.loadmat(boot_ratio_path, variable_names=['bsrs1'])['bsrs1'][:, col]

    weights = pd.DataFrame({
        'Variable': semantic_visual_features,
        'u1_raw': u1,
        'u1': -1 * u1,
        'boot_ratio': boot_ratio,
    })
    weights['p_value'] = 2 * (1 - norm.cdf(abs(weights['boot_ratio'])))
    weights['p_value_adjusted'] = multipletests(weights['p_value'], method='fdr_bh')[1]
    weights['significant'] = weights['p_value_adjusted'] < .05
    return weights


def apply_e1_plot_style(ax, ylabel):
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontproperties=prop, fontsize=20)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(prop)
        lbl.set_fontsize(14)
        lbl.set_rotation(20)
        lbl.set_ha('right')
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


def add_weighted_semantic_condition(df, subject_scores):
    return df.merge(
        subject_scores[['Subnum', 'weighted_semantic_score', group_col]],
        on='Subnum',
        how='left'
    )


def summarize_trial_metric(data, metric):
    metric_data = data.copy()
    if metric in ['rank_2', 'EV_history']:
        metric_data = metric_data[metric_data['exploration'] == 1].copy()

    summary = (
        metric_data[metric_data[group_col].isin(['Low Weighted Semantic', 'High Weighted Semantic'])]
        .groupby(['Subnum', group_col, 'Task'], observed=True)[metric]
        .mean()
        .dropna()
        .reset_index()
    )
    summary[group_col] = pd.Categorical(summary[group_col], categories=condition_order, ordered=True)
    summary['Task'] = summary['Task'].map({1: 'First', 2: 'Second'})
    summary['Task'] = pd.Categorical(summary['Task'], categories=task_order, ordered=True)
    return summary


def plot_metric(summary, metric, ylabel, filename):
    coefficient_summary = (
        summary.groupby([group_col, 'Task'], observed=True)[metric]
        .agg(['mean', 'sem'])
        .reset_index()
    )
    coefficient_summary['se'] = coefficient_summary['sem'].fillna(0)

    fig, ax = plt.subplots(figsize=(5.5, 6.5))
    x_levels = ['Low Weighted Semantic', 'High Weighted Semantic']
    x_positions = {condition: idx for idx, condition in enumerate(x_levels)}
    task_offsets = {'First': -0.14, 'Second': 0.14}

    for _, row in coefficient_summary.iterrows():
        condition = row[group_col]
        task = row['Task']
        if condition not in x_positions or pd.isna(task):
            continue
        ax.errorbar(
            x_positions[condition] + task_offsets[task],
            row['mean'],
            yerr=row['se'],
            fmt='o',
            markersize=12,
            color='black',
            ecolor='black',
            elinewidth=2.5,
            capsize=5,
            markerfacecolor=task_palette[task],
            markeredgecolor='black',
            markeredgewidth=2,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            linestyle='',
            markerfacecolor=task_palette[task],
            markeredgecolor='black',
            markeredgewidth=1.5,
            color='black',
            markersize=11,
            label=task,
        )
        for task in task_order
    ]
    ax.legend(handles=handles, title='Task', loc='lower left')
    ax.axhline(50 if metric == 'Reward' else 0, color='gray', linestyle='--', linewidth=1.8)
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(['1st Quartile', '4th Quartile'])
    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    apply_e1_plot_style(ax, ylabel)
    plt.savefig(f'./figures/{filename}', dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./analysis_outputs', exist_ok=True)

    model_semantic_weights = get_model_semantic_weights(lv=1)
    selected_weights = model_semantic_weights[model_semantic_weights['Variable'].isin(target_features)].copy()
    selected_weights = selected_weights.set_index('Variable').loc[target_features].reset_index()
    selected_weights.to_csv('./analysis_outputs/pls_model_semantic_grass_water_river_weights.csv', index=False)
    print('Selected PLS_model~semantic weights:')
    print(selected_weights[['Variable', 'u1', 'boot_ratio', 'p_value_adjusted', 'significant']])

    dm_switch = pd.read_csv('./data/dm_switch.csv')
    subject_scores = (
        dm_switch[['Subnum', 'Condition'] + target_features]
        .drop_duplicates(subset=['Subnum'])
        .dropna(subset=target_features)
        .copy()
    )

    weight_map = selected_weights.set_index('Variable')['u1']
    for feature in target_features:
        subject_scores[f'{feature}_weighted'] = subject_scores[feature] * weight_map[feature]
    subject_scores['weighted_semantic_score'] = subject_scores[[f'{feature}_weighted' for feature in target_features]].sum(axis=1)

    q1 = subject_scores['weighted_semantic_score'].quantile(.25)
    q3 = subject_scores['weighted_semantic_score'].quantile(.75)
    subject_scores[group_col] = 'Middle Weighted Semantic'
    subject_scores.loc[subject_scores['weighted_semantic_score'] <= q1, group_col] = 'Low Weighted Semantic'
    subject_scores.loc[subject_scores['weighted_semantic_score'] >= q3, group_col] = 'High Weighted Semantic'
    subject_scores[group_col] = pd.Categorical(subject_scores[group_col], categories=condition_order, ordered=True)
    subject_scores.to_csv('./analysis_outputs/weighted_semantic_subject_quartiles.csv', index=False)

    counts = (
        subject_scores
        .groupby([group_col, 'Condition'], observed=False)['Subnum']
        .nunique()
        .reset_index(name='n_participants')
    )
    counts.to_csv('./analysis_outputs/weighted_semantic_quartile_counts.csv', index=False)
    print('\nParticipant counts by weighted semantic condition and original condition:')
    print(counts.to_string(index=False))

    dm_switch = add_weighted_semantic_condition(dm_switch, subject_scores)
    dm_switch.to_csv('./analysis_outputs/dm_switch_weighted_semantic_condition.csv', index=False)

    e1_metrics = {
        'Reward': ('Reward', 'WeightedSemantic_Reward_by_Task.png'),
        'BestChoice': ('P(Optimal Choice)', 'WeightedSemantic_BestChoice_by_Task.png'),
        'value_gap': (r'$\Delta$ Best-Chosen Value', 'WeightedSemantic_Value_Gap_by_Task.png'),
        'Switch': ('P(Switch)', 'WeightedSemantic_Switch_by_Task.png'),
        'WinStay': ('P(Win-Stay)', 'WeightedSemantic_WinStay_by_Task.png'),
        'LoseShift': ('P(Lose-Shift)', 'WeightedSemantic_LoseShift_by_Task.png'),
        'exploration': ('P(Exploration)', 'WeightedSemantic_Exploration_by_Task.png'),
        'rank_2': ('P(Exploratory Second-Best Choice)', 'WeightedSemantic_Rank_2_by_Task.png'),
        'EV_history': ('Exploratory EV Chosen', 'WeightedSemantic_EV_History_by_Task.png'),
    }

    plot_summaries = []
    for metric, (ylabel, filename) in e1_metrics.items():
        summary = summarize_trial_metric(dm_switch, metric)
        summary['Metric'] = metric
        plot_summaries.append(summary.rename(columns={metric: 'value'}))
        plot_metric(summary, metric, ylabel, filename)

    pd.concat(plot_summaries, ignore_index=True).to_csv(
        './analysis_outputs/weighted_semantic_quartile_plot_summaries.csv',
        index=False
    )

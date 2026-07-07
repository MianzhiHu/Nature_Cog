import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
from utils.ComputationalModeling import residual_calculator
import matplotlib.pyplot as plt
import seaborn as sns
import functools
from scipy.stats import pearsonr


# ======================================================================================================================
# Define variable lists
# ======================================================================================================================
identity_cols = ['Subnum', 'Condition', 'Task']
ratings = ['naturalness', 'disorderliness', 'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery',
           'imagability', 'control']
behav_perf = ['BestChoice', 'Reward', 'value_gap', 'Switch', 'WinStay', 'LoseShift']
low_visual_features = ['Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton', 'SDSat', 'Contrast', 'Dissimilarity',
                       'Homogeneity', 'Energy', 'Correlation', 'MeanTexture', 'SDTexture', 'Entropy', 'EdgeCount',
                       'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength', 'ContourSDLength',
                       'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH']
# semantic_visual_features = ['sky', 'grass', 'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench', 'pole',
#                             'building', 'tree', 'earth', 'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair',
#                             'signboard', 'stairs', 'pot', 'sculpture', 'sidewalk', 'railing', 'road', 'person',
#                             'mountain', 'lake', 'floor', 'car', 'traffic light']
semantic_visual_features = ['sky', 'grass', 'plant', 'water', 'fence', 'path', 'river', 'bench', 'pole', 'building',
                            'tree', 'earth', 'rock', 'streetlight', 'wall', 'signboard', 'sidewalk', 'railing', 'road',
                            'person', 'mountain']
semantic_visual_features_freq = [f'{feature}_freq' for feature in semantic_visual_features]
# semantic_visual_features = semantic_visual_features + semantic_visual_features_freq
semantic_pc_features = ['Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']
visual_features = low_visual_features + semantic_visual_features + semantic_pc_features
model_param = ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center', 'Exploration_Rate', 'rank_2_exploration_rate', 'EV_history_exploration']
all_behav = ['Reward', 'BestChoice', 'value_gap', 'Switch', 'WinStay', 'LoseShift', 't', 'dis_sd', 'noise_sd',
             'decay', 'decay_center', 'Exploration_Rate', 'rank_2_exploration_rate', 'EV_history_exploration']


if __name__ == '__main__':
    # ======================================================================================================================
    # Read PLS data
    # ======================================================================================================================
    method = 'difference'

    # Load data
    E1_dm_summary = pd.read_csv('./data/E1_dm_summary.csv')
    E1_model_summary = pd.read_csv(('./data/dm_summary_modeled.csv'))
    E1_freq_summary = pd.read_csv('./data/E1_freq_rating.csv')

    # Change column names in E1_freq_summary to match semantic_visual_features
    E1_freq_summary.rename(columns={feature: f'{feature}_freq' for feature in semantic_visual_features
                                    if not feature.endswith('_freq')}, inplace=True)
    E1_freq_summary = E1_freq_summary[['Subnum'] + semantic_visual_features_freq]
    E1_dm_summary = pd.merge(E1_dm_summary, E1_freq_summary, on='Subnum')

    E1_dm_summary = E1_dm_summary[identity_cols + behav_perf + visual_features + ratings]
    E1_model_summary = E1_model_summary[identity_cols + model_param]

    summary_all = pd.merge(E1_dm_summary, E1_model_summary, on=identity_cols)
    for condition in summary_all['Condition'].unique():
        for task in summary_all['Task'].unique():
            subset = summary_all[(summary_all['Condition'] == condition) & (summary_all['Task'] == task)]
            # correlation between reward and exploration
            reward = subset['Reward']
            exploration = subset['Exploration_Rate']
            print(f'{condition} - Task {task} - Reward vs Exploration: {pearsonr(reward, exploration)}')

    residual = residual_calculator(summary_all, behav_perf + model_param, task1_name=1, task2_name=2, subj_col='Subnum',
                                   task_col='Task', method=method)

    # Parse the data
    condition_list = residual['Condition'].unique().tolist()
    condition_list_no_control = [cond for cond in condition_list if cond != 'Control']
    behav_perf_residual = [perf + '_' + method for perf in behav_perf]
    model_param_residual = [param + '_' + method for param in model_param]
    all_behav_residual = [param + '_' + method for param in all_behav]

    # if behav_perf_residual is NaN, then drop the row
    residual = residual.dropna(subset=all_behav_residual)
    # residual = residual[residual['Condition'] != 'Control'].copy()

    # Z-score the data for all columns
    cols_to_zscore = residual.columns.difference(identity_cols)
    residual[cols_to_zscore] = residual[cols_to_zscore].apply(lambda x: (x - x.mean()) / x.std())

    # Save all data
    pls_sem = residual[residual['Condition'] != 'Control'].copy()
    print(pls_sem.shape)
    print(pls_sem.columns)
    pls_sem.to_csv('./data/PLS_Data/PLS_Sem_E1.csv', index=False)
    pls_sem[ratings].to_csv('./data/PLS_Data/PLS_Ratings_NatureUrban.csv', index=False)
    pls_sem[behav_perf_residual].to_csv('./data/PLS_Data/PLS_BehavPerf_NatureUrban.csv', index=False)
    pls_sem[low_visual_features].to_csv('./data/PLS_Data/PLS_VisualFeatures_NatureUrban.csv', index=False)
    pls_sem[semantic_visual_features].to_csv('./data/PLS_Data/PLS_Semantic_NatureUrban.csv', index=False)
    pls_sem[semantic_pc_features].to_csv('./data/PLS_Data/PLS_SemanticPC_NatureUrban.csv', index=False)
    pls_sem[model_param_residual].to_csv('./data/PLS_Data/PLS_ModelParams_NatureUrban.csv', index=False)

    for cond in condition_list:
        subset = residual[residual['Condition'] == cond].copy()
        ratings_df = subset[ratings].copy()
        behav_perf_df = subset[behav_perf_residual].copy()
        low_visual_features_df = subset[low_visual_features].copy()
        semantic_visual_features_df = subset[semantic_visual_features].copy()
        semantic_pc_features_df = subset[semantic_pc_features].copy()
        visual_features_df = subset[visual_features].copy()
        model_param_df = subset[model_param_residual].copy()
        all_behav_df = subset[all_behav_residual].copy()


        # save to csv
        ratings_df.to_csv(f'./data/PLS_Data/PLS_Ratings_{cond}.csv', index=False)
        behav_perf_df.to_csv(f'./data/PLS_Data/PLS_BehavPerf_{cond}.csv', index=False)
        low_visual_features_df.to_csv(f'./data/PLS_Data/PLS_VisualFeatures_{cond}.csv', index=False)
        semantic_visual_features_df.to_csv(f'./data/PLS_Data/PLS_Semantic_{cond}.csv', index=False)
        semantic_pc_features_df.to_csv(f'./data/PLS_Data/PLS_SemanticPC_{cond}.csv', index=False)
        model_param_df.to_csv(f'./data/PLS_Data/PLS_ModelParams_{cond}.csv', index=False)
        all_behav_df.to_csv(f'./data/PLS_Data/PLS_AllBehav_{cond}.csv', index=False)


        # # print the mean of each semantic column
        # print(f'{cond} - Model Parameter Mean:\n{model_param_df.mean()}\n')
        # print(f'{cond} - Semantic Visual Features Mean:\n{semantic_visual_features_df.mean()}\n')
        # grass = semantic_visual_features_df['grass']
        # exploration = model_param_df['Exploration_Rate_' + method]

        # print (pearsonr(grass, exploration))

    # ==================================================================================================================
    # E2 PLS data
    # ==================================================================================================================
    E2_dm_summary = pd.read_csv('./data/E2_dm_summary.csv')
    E2_model_summary = pd.read_csv('./data/E2_dm_summary_modeled.csv')
    E2_freq_summary = pd.read_csv('./data/E2_freq_rating.csv')

    E2_freq_summary.rename(columns={feature: f'{feature}_freq' for feature in semantic_visual_features
                                    if not feature.endswith('_freq')}, inplace=True)
    E2_freq_summary = E2_freq_summary[['Subnum'] + semantic_visual_features_freq]
    E2_dm_summary = pd.merge(E2_dm_summary, E2_freq_summary, on='Subnum')

    E2_dm_summary = E2_dm_summary[identity_cols + behav_perf + visual_features + ratings]
    E2_model_summary = E2_model_summary[identity_cols + model_param]
    E2_summary_all = pd.merge(E2_dm_summary, E2_model_summary, on=identity_cols)
    E2_summary_all = E2_summary_all.dropna(subset=behav_perf)

    cols_to_zscore = E2_summary_all.columns.difference(identity_cols)
    E2_summary_all[cols_to_zscore] = E2_summary_all[cols_to_zscore].apply(lambda x: (x - x.mean()) / x.std())

    E2_pls_sem = E2_summary_all[E2_summary_all['Condition'] != 'Control'].copy()
    print(E2_pls_sem.shape)
    print(E2_pls_sem.columns)
    E2_pls_sem.to_csv('./data/PLS_Data/PLS_Sem_E2.csv', index=False)
    E2_pls_sem[ratings].to_csv('./data/PLS_Data/PLS_Ratings_E2_NatureUrban.csv', index=False)
    E2_pls_sem[behav_perf].to_csv('./data/PLS_Data/PLS_BehavPerf_E2_NatureUrban.csv', index=False)
    E2_pls_sem[low_visual_features].to_csv('./data/PLS_Data/PLS_VisualFeatures_E2_NatureUrban.csv', index=False)
    E2_pls_sem[semantic_visual_features].to_csv('./data/PLS_Data/PLS_Semantic_E2_NatureUrban.csv', index=False)
    E2_pls_sem[semantic_pc_features].to_csv('./data/PLS_Data/PLS_SemanticPC_E2_NatureUrban.csv', index=False)
    E2_pls_sem[model_param].to_csv('./data/PLS_Data/PLS_ModelParams_E2_NatureUrban.csv', index=False)
    E2_pls_sem[all_behav].to_csv('./data/PLS_Data/PLS_AllBehav_E2_NatureUrban.csv', index=False)

    for cond in E2_summary_all['Condition'].unique():
        subset = E2_summary_all[E2_summary_all['Condition'] == cond].copy()
        ratings_df = subset[ratings].copy()
        behav_perf_df = subset[behav_perf].copy()
        low_visual_features_df = subset[low_visual_features].copy()
        semantic_visual_features_df = subset[semantic_visual_features].copy()
        semantic_pc_features_df = subset[semantic_pc_features].copy()
        model_param_df = subset[model_param].copy()
        all_behav_df = subset[all_behav].copy()

        ratings_df.to_csv(f'./data/PLS_Data/PLS_Ratings_E2_{cond}.csv', index=False)
        behav_perf_df.to_csv(f'./data/PLS_Data/PLS_BehavPerf_E2_{cond}.csv', index=False)
        low_visual_features_df.to_csv(f'./data/PLS_Data/PLS_VisualFeatures_E2_{cond}.csv', index=False)
        semantic_visual_features_df.to_csv(f'./data/PLS_Data/PLS_Semantic_E2_{cond}.csv', index=False)
        semantic_pc_features_df.to_csv(f'./data/PLS_Data/PLS_SemanticPC_E2_{cond}.csv', index=False)
        model_param_df.to_csv(f'./data/PLS_Data/PLS_ModelParams_E2_{cond}.csv', index=False)
        all_behav_df.to_csv(f'./data/PLS_Data/PLS_AllBehav_E2_{cond}.csv', index=False)


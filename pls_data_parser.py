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
           'imageability', 'control']
behav_perf = ['BestChoice', 'Reward', 'Switch', 'WinStay', 'LoseShift']
behav_perf_residual = ['BestChoice_resid', 'Reward_resid', 'Switch_resid', 'WinStay_resid', 'LoseShift_resid']
quad_residual = ['BestChoice_2nd_Quadratic', 'Reward_2nd_Quadratic', 'Switch_2nd_Quadratic', 'WinStay_2nd_Quadratic',
                 'LoseShift_2nd_Quadratic', 'Exploration_2nd_Quadratic']
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
model_param = ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center', 'Exploration_Rate']
model_param_residual = ['t_resid', 'dis_sd_resid', 'noise_sd_resid', 'decay_resid', 'decay_center_resid', 'exploration_resid']


if __name__ == '__main__':
    # ======================================================================================================================
    # Read PLS data
    # ======================================================================================================================
    method = 'difference'
    
    # Load data
    E1_dm_summary = pd.read_csv('./data/E1_dm_summary.csv')
    model_summary = pd.read_csv(('./data/dm_summary_modeled.csv'))
    E1_residual_summary = pd.read_csv('./data/behavior_residuals.csv')
    E1_freq_summary = pd.read_csv('./data/E1_freq_rating.csv')

    # Change column names in E1_freq_summary to match semantic_visual_features
    E1_freq_summary.rename(columns={feature: f'{feature}_freq' for feature in semantic_visual_features
                                    if not feature.endswith('_freq')}, inplace=True)
    E1_freq_summary = E1_freq_summary[['Subnum'] + semantic_visual_features_freq]
    E1_dm_summary = pd.merge(E1_dm_summary, E1_freq_summary, on='Subnum')
   
    # E1_dm_summary = E1_dm_summary[
    #     identity_cols + [x for x in behav_perf if x != 'Exploration_Rate'] + visual_features + ratings]
    # model_summary = model_summary[identity_cols + model_param + ['Exploration_Rate']]

    E1_dm_summary = E1_dm_summary[identity_cols + behav_perf + visual_features + ratings]
    model_summary = model_summary[identity_cols + model_param]
    E1_residual_summary = E1_residual_summary[['Subnum'] + behav_perf_residual + quad_residual + model_param_residual]

    summary_all = pd.merge(E1_dm_summary, model_summary, on=identity_cols)
    for condition in summary_all['Condition'].unique():
        for task in summary_all['Task'].unique():
            subset = summary_all[(summary_all['Condition'] == condition) & (summary_all['Task'] == task)]
            # correlation between reward and exploration
            reward = subset['Reward']
            exploration = subset['Exploration_Rate']
            print(f'{condition} - Task {task} - Reward vs Exploration: {pearsonr(reward, exploration)}')

    summary_all = pd.merge(summary_all, E1_residual_summary, on='Subnum')

    residual = residual_calculator(summary_all, behav_perf + model_param, task1_name=1, task2_name=2, subj_col='Subnum',
                                   task_col='Task', method=method)

    # Parse the data
    condition_list = residual['Condition'].unique().tolist()
    condition_list_no_control = [cond for cond in condition_list if cond != 'Control']
    behav_perf = [perf + '_' + method for perf in behav_perf]
    model_param = [param + '_' + method for param in model_param]

    # if behav_perf_residual is NaN, then drop the row
    residual = residual.dropna(subset=behav_perf)
    # residual = residual[residual['Condition'] != 'Control'].copy()

    # Z-score the data for all columns
    cols_to_zscore = residual.columns.difference(identity_cols)
    residual[cols_to_zscore] = residual[cols_to_zscore].apply(lambda x: (x - x.mean()) / x.std())

    # Save all data
    pls_sem = residual[residual['Condition'] != 'Control'].copy()
    print(pls_sem.shape)
    print(pls_sem.columns)
    pls_sem.to_csv('./data/PLS_Data/PLS_Sem_E1.csv', index=False)

    for cond in condition_list:
        subset = residual[residual['Condition'] == cond].copy()
        ratings_df = subset[ratings].copy()
        behav_perf_df = subset[behav_perf].copy()
        behav_perf_residual_df = subset[behav_perf_residual].copy()
        quad_df = subset[quad_residual].copy()
        low_visual_features_df = subset[low_visual_features].copy()
        semantic_visual_features_df = subset[semantic_visual_features].copy()
        semantic_pc_features_df = subset[semantic_pc_features].copy()
        visual_features_df = subset[visual_features].copy()
        model_param_df = subset[model_param].copy()
        model_param_residual_df = subset[model_param_residual].copy()

        # save to csv
        ratings_df.to_csv(f'./data/PLS_Data/PLS_Ratings_{cond}.csv', index=False)
        behav_perf_df.to_csv(f'./data/PLS_Data/PLS_BehavPerf_{cond}.csv', index=False)
        behav_perf_residual_df.to_csv(f'./data/PLS_Data/PLS_BehavPerfResidual_{cond}.csv', index=False)
        quad_df.to_csv(f'./data/PLS_Data/PLS_BehavPerfQuadratic_{cond}.csv', index=False)
        low_visual_features_df.to_csv(f'./data/PLS_Data/PLS_VisualFeatures_{cond}.csv', index=False)
        semantic_visual_features_df.to_csv(f'./data/PLS_Data/PLS_Semantic_{cond}.csv', index=False)
        semantic_pc_features_df.to_csv(f'./data/PLS_Data/PLS_SemanticPC_{cond}.csv', index=False)
        model_param_df.to_csv(f'./data/PLS_Data/PLS_ModelParams_{cond}.csv', index=False)
        model_param_residual_df.to_csv(f'./data/PLS_Data/PLS_ModelParamsResidual_{cond}.csv', index=False)


        # print the mean of each semantic column
        print(f'{cond} - Model Parameter Mean:\n{model_param_df.mean()}\n')
        print(f'{cond} - Semantic Visual Features Mean:\n{semantic_visual_features_df.mean()}\n')
        grass = semantic_visual_features_df['grass']
        exploration = model_param_residual_df['exploration_resid']
        # exploration = model_param_df['Exploration_Rate_' + method]

        print (pearsonr(grass, exploration))



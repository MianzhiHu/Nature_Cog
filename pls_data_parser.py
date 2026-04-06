import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
from utils.ComputationalModeling import residual_calculator
import matplotlib.pyplot as plt
import seaborn as sns
import functools


# ======================================================================================================================
# Define variable lists
# ======================================================================================================================
identity_cols = ['Subnum', 'Condition', 'Task']
ratings = ['naturalness', 'disorderliness', 'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery',
           'imagability', 'control']
behav_perf = ['BestChoice', 'Reward', 'Switch', 'WinStay', 'LoseShift']
low_visual_features = ['Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton', 'SDSat', 'Contrast', 'Dissimilarity',
                       'Homogeneity', 'Energy', 'Correlation', 'MeanTexture', 'SDTexture', 'Entropy', 'EdgeCount',
                       'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength', 'ContourSDLength',
                       'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH']
semantic_visual_features = ['sky', 'grass', 'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench', 'pole',
                            'building', 'tree', 'earth', 'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair',
                            'signboard', 'stairs', 'pot', 'sculpture', 'sidewalk', 'railing', 'road', 'person',
                            'mountain', 'lake', 'floor', 'car', 'traffic light']
# semantic_visual_features = ['sky', 'grass', 'plant', 'water', 'fence', 'path', 'river', 'bench', 'pole', 'building',
#                             'tree', 'earth', 'rock', 'streetlight', 'wall', 'signboard', 'sidewalk', 'railing', 'road',
#                             'person', 'mountain', 'car']
visual_features = low_visual_features + semantic_visual_features
model_param = ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center', 'Exploration_Rate']


if __name__ == '__main__':
    # ======================================================================================================================
    # Read PLS data
    # ======================================================================================================================
    # Load data
    E1_dm_summary = pd.read_csv('./data/E1_dm_summary.csv')
    model_summary = pd.read_csv(('./data/dm_summary_modeled.csv'))
    method = 'residual'

    # E1_dm_summary = E1_dm_summary[
    #     identity_cols + [x for x in behav_perf if x != 'Exploration_Rate'] + visual_features + ratings]
    # model_summary = model_summary[identity_cols + model_param + ['Exploration_Rate']]

    E1_dm_summary = E1_dm_summary[identity_cols + behav_perf + visual_features + ratings]
    model_summary = model_summary[identity_cols + model_param]

    E1_overlap_cols = set(E1_dm_summary.columns).intersection(set(model_summary.columns))
    summary_all = pd.merge(E1_dm_summary, model_summary, on=identity_cols)

    residual = residual_calculator(summary_all, behav_perf + model_param, task1_name=1, task2_name=2, subj_col='Subnum',
                                   task_col='Task', method=method)

    # Save all data
    pls_sem = residual[residual['Condition'] != 'Control'].copy()
    print(pls_sem.shape)
    print(pls_sem.columns)
    pls_sem.to_csv('./data/PLS_Data/PLS_Sem_IGT_SGT.csv', index=False)

    # Parse the data
    condition_list = residual['Condition'].unique().tolist()
    behav_perf_residual = [perf + '_' + method for perf in behav_perf]
    model_param_residual = [param + '_' + method for param in model_param]

    # if behav_perf_residual is NaN, then drop the row
    residual = residual.dropna(subset=behav_perf_residual)

    for cond in condition_list:
        subset = residual[residual['Condition'] == cond].copy()
        ratings_df = subset[ratings].copy()
        behav_perf_df = subset[behav_perf_residual].copy()
        low_visual_features_df = subset[low_visual_features].copy()
        semantic_visual_features_df = subset[semantic_visual_features].copy()
        visual_features_df = subset[visual_features].copy()
        model_param_df = subset[model_param_residual].copy()

        # save to csv
        ratings_df.to_csv(f'./data/PLS_Data/PLS_Ratings_{cond}.csv', index=False)
        behav_perf_df.to_csv(f'./data/PLS_Data/PLS_BehavPerf_{cond}.csv', index=False)
        low_visual_features_df.to_csv(f'./data/PLS_Data/PLS_VisualFeatures_{cond}.csv', index=False)
        semantic_visual_features_df.to_csv(f'./data/PLS_Data/PLS_Semantic_{cond}.csv', index=False)
        model_param_df.to_csv(f'./data/PLS_Data/PLS_ModelParams_{cond}.csv', index=False)



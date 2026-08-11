import sys
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from matplotlib.pyplot import ylabel
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pls_data_parser import (behav_perf, visual_features, model_param, semantic_visual_features, low_visual_features,
                             ratings, semantic_pc_features)
from matplotlib import font_manager as fm
from utils.PLS_Reader import plot_predictor_results, plot_outcome_results

# add path to the PLS results
result_dir = os.path.abspath('C:/Users/User/OneDrive/Desktop/胡勉之/Texas A&M University/PLS/Result/Nature_Cog/')
ori_data = pd.read_csv('./data/PLS_Data/PLS_Sem_E1.csv')
sys.path.append(result_dir)

def get_pls_results(lv_path, boot_ratio_path, var_names, method='fdr_bh', p=.05, LV=1,
                    anchor_variable=None):
    """Read one PLS LV and impose a deterministic, shared sign convention.

    By default, the predictor with the largest absolute weight is oriented
    positive. Pass ``anchor_variable`` to use an explicit positive anchor.
    Reuse ``df.attrs['orientation']`` when plotting this LV's outcomes.
    """

    # define the path
    lv_path = os.path.join(result_dir, lv_path)
    boot_ratio_path = os.path.join(result_dir, boot_ratio_path)
    col = LV - 1

    # read lv_vals file
    lv_vals = sio.loadmat(lv_path, variable_names=['u1'])

    # read bootstrap ratio file
    boot_ratio = sio.loadmat(boot_ratio_path, variable_names=['bsrs1'])

    u1 = lv_vals['u1'][:, col]
    boot_ratio = boot_ratio['bsrs1'][:, col]

    if len(u1) != len(var_names):
        raise ValueError(
            f'LV has {len(u1)} weights, but {len(var_names)} names were supplied.'
        )
    if anchor_variable is None:
        if not np.isfinite(u1).any():
            raise ValueError('Cannot orient an LV with no finite weights.')
        anchor_idx = int(np.nanargmax(np.abs(u1)))
    else:
        try:
            anchor_idx = list(var_names).index(anchor_variable)
        except ValueError as exc:
            raise ValueError(f'Unknown anchor variable: {anchor_variable}') from exc
        if not np.isfinite(u1[anchor_idx]) or np.isclose(u1[anchor_idx], 0):
            raise ValueError(f'Anchor {anchor_variable} has a zero or non-finite weight.')

    orientation = 1 if u1[anchor_idx] >= 0 else -1
    u1 = orientation * u1
    boot_ratio = orientation * boot_ratio

    # combine the data with their respective columns
    result = np.column_stack((u1, boot_ratio))

    # name the columns
    df = pd.DataFrame(result, columns=['u1', 'boot_ratio'])

    # calculate the p values according to the boot_ratio
    df['p_value'] = 2 * (1 - norm.cdf(abs(df['boot_ratio'])))

    # adjust the p values for multiple comparisons
    df['p_value_adjusted'] = multipletests(df['p_value'], method=method)[1]

    # if boot_ratio is greater than 1.96, then the corresponding u1 value is significant
    df['significant'] = abs(df['p_value_adjusted']) < p

    # add the variable names to the DataFrame as the first column
    df.insert(0, 'Variable', var_names)
    df.attrs['orientation'] = orientation
    df.attrs['anchor_variable'] = list(var_names)[anchor_idx]

    return df

#
# behav_visual_results = get_pls_results('PLS_behav~visual_lv_vals.mat',
#                                         'PLS_behav~visual.mat',
#                                         low_visual_features, method='fdr_bh', p=.05)
#
# behav_ratings_results = get_pls_results('PLS_behav~ratings_lv_vals.mat',
#                                         'PLS_behav~ratings.mat',
#                                         ratings, method='fdr_bh', p=.05)
#
model_ratings_results = get_pls_results('PLS_model~ratings_lv_vals.mat',
                                        'PLS_model~ratings.mat',
                                        ratings, method='fdr_bh', p=.05)
model_ratings_results = get_pls_results('PLS_model~ratingsE2_lv_vals.mat',
                                        'PLS_model~ratingsE2.mat',
                                        ratings, method='fdr_bh', p=.05)
#
#
# model_visual_results = get_pls_results('PLS_model~visual_lv_vals.mat',
#                                         'PLS_model~visual.mat',
#                                         low_visual_features, method='fdr_bh', p=.05)
#
#
model_semantic_results = get_pls_results('PLS_model~semantic_lv_vals.mat',
                                        'PLS_model~semantic.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05)

model_semantic_results = get_pls_results('PLS_model~semanticE2_lv_vals.mat',
                                        'PLS_model~semanticE2.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05)


modelparam_semantic_results = get_pls_results('PLS_modelparam~semanticE2_lv_vals.mat',
                                        'PLS_modelparam~semanticE2.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05)
#
# behav_semantic_results = get_pls_results('PLS_behav~semantic_lv_vals.mat',
#                                         'PLS_behav~semantic.mat',
#                                         semantic_visual_features, method='fdr_bh', p=.05)
#
# behav_semanticpc_results = get_pls_results('PLS_behav~semanticpc_lv_vals.mat',
#                                         'PLS_behav~semanticpc.mat',
#                                         semantic_pc_features, method='fdr_bh', p=.05)
#
ratings_semantic_results = get_pls_results('PLS_ratings~semantics_lv_vals.mat',
                                        'PLS_ratings~semantics.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05)
#
#
# u1_df = model_semantic_results
# name_col = u1_df.columns[0]
# u1_col = u1_df.columns[1]
#
# u1_map = u1_df.set_index(name_col)[u1_col]
# target_cols = pd.Index(semantic_visual_features).intersection(ori_data.columns)
#
# # keep only numeric target columns
# num_cols = ori_data[target_cols].select_dtypes(include='number').columns
#
# # align factors to those columns
# factors = u1_map.reindex(num_cols).fillna(1)
# scaled_df = ori_data.copy()
# scaled_df[num_cols] = scaled_df[num_cols].mul(factors, axis=1)
#
# scaled_df['all'] = scaled_df[num_cols].sum(axis=1)
# print(scaled_df.groupby('Condition')['all'].mean())
# print(ori_data.groupby('Condition')['Exploration_Rate_difference'].mean())
#
# # Plot
model_param_names = ['Reward', 'Optimal Choice', 'Best-Chosen Value', 'Switch', 'Win-Stay', 'Lose-Shift',
                     'Inverse Temperature', 'Reward Variance', 'Noise Variance', 'Decay Rate', 'Decay Center',
                     'Exploration', 'Second-Best Choice', 'EV Chosen']
model_semantic_fig = plot_predictor_results(model_semantic_results, only_sig=False, save_path='./figures/PLS_Model_Semantic_Significant_Results.png')
model_ratings_fig = plot_predictor_results(model_ratings_results, only_sig=False,
                                           save_path='./figures/PLS_Model_Ratings_Significant_Results.png')
model_ratingsE2_fig = plot_predictor_results(model_ratings_results, only_sig=False,
                                           save_path='./figures/PLS_Model_RatingsE2_Significant_Results.png')
ratings_semantic_fig = plot_predictor_results(ratings_semantic_results, only_sig=False, ylabel='Subjective Rating Loadings',
                                              reverse_sign=True, save_path='./figures/PLS_Ratings_Semantic_Significant_Results.png')



plot_outcome_results(result_dir=result_dir, boot_ratio_path='PLS_model~semanticE2.mat', method=3, ylabel='Correlation with LV',
                         conditions=['Nature', 'Urban'], LV_Vis=1, BehavLabels=model_param_names,
                         title=False, save_path='./figures/')
plot_outcome_results(result_dir=result_dir, boot_ratio_path='PLS_model~ratings.mat', method=3, ylabel='Correlation with LV',
                     conditions=['Nature', 'Urban'], LV_Vis=1, BehavLabels=model_param_names,
                     title=False, save_path='./figures/model~ratings')
plot_outcome_results(result_dir=result_dir, boot_ratio_path='PLS_model~ratingsE2.mat', method=3, ylabel='Correlation with LV',
                     conditions=['Nature', 'Urban'], LV_Vis=1, BehavLabels=model_param_names,
                     title=False, save_path='./figures/model~ratingsE2')
plot_outcome_results(result_dir=result_dir, boot_ratio_path='PLS_ratings~semantics.mat', method=3, ylabel='Correlation with LV',
                     conditions=['Nature'], LV_Vis=1, BehavLabels=ratings, reverse_sign=True,
                     title=False, save_path='./figures/ratings~semantics')

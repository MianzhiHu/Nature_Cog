import numpy as np
import pandas as pd
import pingouin as pg
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import functools


# ======================================================================================================================
# Define variable lists
# ======================================================================================================================
ratings = ['naturalness', 'disorderliness', 'aesthetic']
# behav_perf = ['BestOption_z_Diff', 'HighFreqOption_z_Diff', 'HighMagOption_z_Diff', 'IGT_Deck_A', 'IGT_Deck_B',
#               'IGT_Deck_C', 'IGT_Deck_D', 'SGT_Deck_A', 'SGT_Deck_B', 'SGT_Deck_C', 'SGT_Deck_D']
behav_perf = ['BestOption_z_Diff', 'HighFreqOption_z_Diff', 'HighMagOption_z_Diff']
visual_features = ['Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton', 'SDSat', 'Entropy', 'EdgeCount', 'CornerMean',
                   'CornerSD', 'CornerCount', 'ContourMeanLength', 'ContourSDLength', 'ContourMeanArea', 'ContourSDArea',
                   'ContourCount', 'AsymmetryV', 'AsymmetryH', 'KPMeanSize', 'KPSDSize', 'KPMeanStrength', 'KPSDStrength',
                   'KPMeanAngle', 'KPSDAngle', 'KPCount']
model_param = ['alpha_z_IGT', 'alpha_z_SGT', 'la_z_IGT', 'la_z_SGT', 'shape_z_IGT', 'shape_z_SGT', 't_z_IGT', 't_z_SGT', 't_Diff_z',
               'alpha_Diff_z', 'shape_Diff_z', 'la_Diff_z']
dm_summary = pd.read_csv('./data/dm_summary.csv')
dm_summary_wide = pd.read_csv(('./data/dm_summary_task_wide.csv'))
deck_summary = pd.read_csv(('./data/deck_summary.csv'))
model_summary = pd.read_csv(('./data/dm_summary_modeled_wide.csv'))
dm_summary = dm_summary[['Subnum', 'Condition', 'Order'] + ratings + visual_features].drop_duplicates()

if __name__ == '__main__':
    # ======================================================================================================================
    # Read PLS data
    # ======================================================================================================================
    # pivot the deck summary to wide format
    deck_summary_wide = deck_summary.pivot_table(index=['Subnum', 'Condition', 'Order'], columns=['Task', 'keyResponse'], values='ChoiceRate_z').reset_index()
    deck_summary_wide.columns = ['_'.join(map(str, col)).strip() if col[1] else col[0] for col in deck_summary_wide.columns.values]

    # combine all data into one dataframe
    print(f'Shape: dm_summary_wide: {dm_summary_wide.shape}, dm_summary: {dm_summary.shape}, deck_summary_wide: {deck_summary_wide.shape}')
    summary_all = functools.reduce(lambda left, right: pd.merge(left, right, on=['Subnum', 'Condition', 'Order'], how='left'),
                                  [dm_summary_wide, deck_summary_wide, model_summary, dm_summary])
    print(summary_all.shape)

    is_pls_sem = summary_all[(summary_all['Order'] == 'IGT_SGT') &
                             (summary_all['Condition'] != 'Control')].copy()
    print(is_pls_sem.shape)
    print(is_pls_sem.columns)
    is_pls_sem.to_csv('./data/PLS_Data/PLS_Sem_IGT_SGT.csv', index=False)

    # Parse the data
    condition_list = summary_all['Condition'].unique().tolist()
    if 'Control' in condition_list:
        condition_list.remove('Control') # remove control condition because we do not have visual ratings for control condition
    order_list = summary_all['Order'].unique().tolist()

    for cond in condition_list:
        for order in order_list:
            if order == 'IGT_SGT':
                # Remove IGT columns
                order_specific_behav_perf = [col for col in behav_perf if not col.startswith('IGT_')]
                order_specific_model_param = [col for col in model_param if not col.endswith('_IGT')]
            else:
                # Remove SGT columns
                order_specific_behav_perf = [col for col in behav_perf if not col.startswith('SGT_')]
                order_specific_model_param = [col for col in model_param if not col.endswith('_SGT')]

            subset = summary_all[(summary_all['Condition'] == cond) & (summary_all['Order'] == order)].copy()
            ratings_df = subset[ratings].copy()
            behav_perf_df = subset[order_specific_behav_perf].copy()
            visual_features_df = subset[visual_features].copy()
            model_param_df = subset[order_specific_model_param].copy()

            # save to csv
            ratings_df.to_csv(f'./data/PLS_Data/PLS_Ratings_{cond}_{order}.csv', index=False)
            behav_perf_df.to_csv(f'./data/PLS_Data/PLS_BehavPerf_{cond}_{order}.csv', index=False)
            visual_features_df.to_csv(f'./data/PLS_Data/PLS_VisualFeatures_{cond}_{order}.csv', index=False)
            model_param_df.to_csv(f'./data/PLS_Data/PLS_ModelParams_{cond}_{order}.csv', index=False)



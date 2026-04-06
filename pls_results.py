import sys
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pls_data_parser import behav_perf, visual_features, model_param, semantic_visual_features, low_visual_features, ratings


# add path to the PLS results
result_dir = os.path.abspath('C:/Users/zuire/OneDrive/桌面/胡勉之/Texas A&M University/PLS/Result/Nature_Cog/')
sys.path.append(result_dir)

def get_pls_results(lv_path, boot_ratio_path, var_names, method='fdr_bh', p=.05, LV=1):

    # define the path
    lv_path = os.path.join(result_dir, lv_path)
    boot_ratio_path = os.path.join(result_dir, boot_ratio_path)
    col = LV - 1

    # read lv_vals file
    lv_vals = sio.loadmat(lv_path)

    # read bootstrap ratio file
    boot_ratio = sio.loadmat(boot_ratio_path)

    u1 = lv_vals['u1'][:, col]
    boot_ratio = boot_ratio['bsrs1'][:, col]

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

    return df


behav_visual_results = get_pls_results('PLS_behav~visual_lv_vals.mat',
                                        'PLS_behav~visual.mat',
                                        low_visual_features, method='fdr_bh', p=.05)

behav_ratings_results = get_pls_results('PLS_behav~ratings_lv_vals.mat',
                                        'PLS_behav~ratings.mat',
                                        ratings, method='fdr_bh', p=.05)

model_ratings_results = get_pls_results('PLS_model~ratings_lv_vals.mat',
                                        'PLS_model~ratings.mat',
                                        ratings, method='fdr_bh', p=.05)

model_visual_results = get_pls_results('PLS_model~visual_lv_vals.mat',
                                        'PLS_model~visual.mat',
                                        low_visual_features, method='fdr_bh', p=.05)


model_semantic_results = get_pls_results('PLS_model~semantic_lv_vals.mat',
                                        'PLS_model~semantic.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05)

behav_semantic_results = get_pls_results('PLS_behav~semantic_lv_vals.mat',
                                        'PLS_behav~semantic.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05, LV=1)

ratings_semantic_results = get_pls_results('PLS_ratings~semantic_lv_vals.mat',
                                        'PLS_ratings~semantic.mat',
                                        semantic_visual_features, method='fdr_bh', p=.05, LV=1)

# extract the visual features components
visual_feature_lv = sio.loadmat(os.path.join(result_dir, 'PLS_behav~visual.mat'))['result']['usc'][0][0][:, 0]
visual_feature_lv = pd.DataFrame(visual_feature_lv, columns=['VisualFeature_LV1'])

# print working directory
dm_summary = pd.read_csv('./data/dm_summary.csv')
dm_summary_wide = pd.read_csv(('./data/dm_summary_task_wide.csv'))
dm_summary_igt_sgt = dm_summary[(dm_summary['Order'] == 'IGT_SGT') & (dm_summary['Condition'] != 'Control') &
                                (dm_summary['Task'] == 'SGT')].copy()
dm_summary_wide_igt_sgt = dm_summary_wide[(dm_summary_wide['Order'] == 'IGT_SGT') & (dm_summary_wide['Condition'] != 'Control')].copy()

# concatenate the visual feature lv with the dm_summary_igt_sgt
dm_summary_igt_sgt = pd.concat([dm_summary_igt_sgt.reset_index(drop=True), visual_feature_lv.reset_index(drop=True)], axis=1)
dm_summary_wide_igt_sgt = pd.concat([dm_summary_wide_igt_sgt.reset_index(drop=True), visual_feature_lv.reset_index(drop=True)], axis=1)

# plot the scatter plot between visual feature lv and BestOption_z_Diff
plt.figure()
sns.regplot(data=dm_summary_igt_sgt, x='VisualFeature_LV1', y='BestOption')
plt.title('Scatter Plot between Visual Feature LV1 and BestOption_z_Diff (IGT-SGT)')
plt.xlabel('Visual Feature LV1')
plt.ylabel('BestOption_z_Diff (SGT - IGT)')
plt.savefig('./figures/Scatter_VisualFeatureLV1_BestOptionzDiff_IGT_SGT.png', dpi=600)
plt.close()
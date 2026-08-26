import pandas as pd
from statsmodels.stats.multitest import multipletests
from pls_data_parser import semantic_visual_features

# Import results
E1_pairwise = pd.read_csv('./data/e1_semantic_difference_model_coefficients_R.csv')
E2_pairwise = pd.read_csv('./data/e2_semantic_feature_model_coefficients_R.csv')
E2_joint_semantic = pd.read_csv('./data/e2_joint_semantic_behavior_model_coefficients_R.csv')
E2_joint_ratings = pd.read_csv('./data/e2_joint_rating_behavior_model_coefficients_R.csv')
E1_trial_ratings = pd.read_csv('./data/e1_trialwise_semantic_rating_model_coefficients_R.csv')

# Keep only feature and interaction effects
E1_pairwise = E1_pairwise[
    ((E1_pairwise['model_type'] == 'additive') & (E1_pairwise['term'] == 'feature_z')) |
    ((E1_pairwise['model_type'] == 'interaction') & (E1_pairwise['term'] == 'feature_z:ConditionUrban'))
].copy()

E2_pairwise = E2_pairwise[
    ((E2_pairwise['model_type'] == 'additive') & (E2_pairwise['term'] == 'feature_z')) |
    ((E2_pairwise['model_type'] == 'interaction') & (E2_pairwise['term'] == 'feature_z:ConditionUrban'))
].copy()

E2_joint_semantic = E2_joint_semantic[
    ((E2_joint_semantic['model_type'] == 'additive') & E2_joint_semantic['term'].str.endswith('_z')) |
    ((E2_joint_semantic['model_type'] == 'interaction') & E2_joint_semantic['term'].str.endswith('_z:ConditionUrban')
     ) &
    (E2_joint_semantic['converged'] == True) &
    (E2_joint_semantic['singular'] == False)
].copy()

E2_joint_ratings = E2_joint_ratings[
    (E2_joint_ratings['model_type'] == 'additive') &
    E2_joint_ratings['term'].str.endswith('_z') &
    (E2_joint_ratings['singular'] == False)
].copy()

E1_trial_rating_effects = E1_trial_ratings[
    (
        ((E1_trial_ratings['model_type'] == 'additive') & E1_trial_ratings['term'].str.endswith('_z')) |
        ((E1_trial_ratings['model_type'] == 'interaction') & E1_trial_ratings['term'].str.endswith('_z:ConditionUrban'))
    ) &
    (E1_trial_ratings['converged'] == True) &
    (E1_trial_ratings['singular'] == False)
].copy()


# Combine t-test and z-test p-values for E2
E2_pairwise['p_value'] = E2_pairwise['Pr(>|t|)'].fillna(E2_pairwise['Pr(>|z|)'])
E2_joint_semantic['p_value'] = E2_joint_semantic['Pr(>|t|)'].fillna(E2_joint_semantic['Pr(>|z|)'])
E2_joint_ratings['p_value'] = E2_joint_ratings['Pr(>|t|)'].fillna(E2_joint_ratings['Pr(>|z|)'])
E1_trial_rating_effects['p_value'] = E1_trial_rating_effects['Pr(>|t|)']


# Correct p-values globally using FDR
E1_pairwise['p_value_adjusted'] = multipletests(E1_pairwise['Pr(>|t|)'], method='fdr_bh')[1]
E2_pairwise['p_value_adjusted'] = multipletests(E2_pairwise['p_value'], method='fdr_bh')[1]
E2_joint_semantic['p_value_adjusted'] = multipletests(E2_joint_semantic['p_value'], method='fdr_bh')[1]
E2_joint_ratings['p_value_adjusted'] = multipletests(E2_joint_ratings['p_value'], method='fdr_bh')[1]
E1_trial_rating_effects['p_value_adjusted_all'] = multipletests(E1_trial_rating_effects['p_value'], method='fdr_bh')[1]

# Sort by adjusted p-value
E2_sig = E2_pairwise[E2_pairwise['p_value_adjusted'] < 0.05]
E1_trial_rating_all_sig = E1_trial_rating_effects[E1_trial_rating_effects['p_value_adjusted_all'] < 0.05]
E1_trial_rating_additive_sig = E1_trial_rating_all_sig[E1_trial_rating_all_sig['model_type'] == 'additive']
E1_trial_rating_interaction_sig = E1_trial_rating_all_sig[E1_trial_rating_all_sig['model_type'] == 'interaction']


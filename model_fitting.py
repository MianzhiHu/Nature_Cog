import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import ruptures as rpt
from utils.ComputationalModeling import (ComputationalModels, dict_generator, moving_window_model_fitting,
                                         parameter_extractor)
from utils.DualProcess import DualProcessModel
import functools

# ======================================================================================================================
# Load the data
# ======================================================================================================================
dm_data = pd.read_csv('./data/dm_data.csv')
dm_summary = pd.read_csv('./data/dm_summary.csv')
# For the purpose of 2025 Psychonomics, we only use IGT-SGT data
dm_data = dm_data[dm_data['Order'] == 'IGT_SGT'].copy()
dm_data['Trial'] = dm_data['Trial'].astype(int)
dm_data['keyResponse'] = dm_data['keyResponse'].astype(int)
dm_data['keyResponse'] = dm_data['keyResponse'] - 1
img_data = pd.read_csv('./data/img_data.csv')
SGT_data = dm_data[dm_data['Task'] == 'SGT'].copy()
IGT_data = dm_data[dm_data['Task'] == 'IGT'].copy()

# ======================================================================================================================
# Data Preprocessing
# ======================================================================================================================
SGT_dict = dict_generator(SGT_data, 'IGT_SGT')
IGT_dict = dict_generator(IGT_data, 'IGT_SGT')

# ======================================================================================================================
# Model Fitting
# ======================================================================================================================
# Define the model parameters
delta = ComputationalModels('delta', task='IGT_SGT')
decay = ComputationalModels('decay', task='IGT_SGT')
delta_PVL = ComputationalModels('delta_PVL', task='IGT_SGT')
decay_PVL = ComputationalModels('decay_PVL', task='IGT_SGT')
delta_asymmetric = ComputationalModels('delta_asymmetric', task='IGT_SGT')
decay_PVPE = ComputationalModels('decay_PVPE', task='IGT_SGT')
# dual_process = DualProcessModel(num_trials=100, task='IGT_SGT', default_EV=0.0)

model_list = [decay_PVL]
model_name_list = [name for name, obj in globals().items() if any(obj is m for m in model_list)]

# Set window parameters
window_size = 10
n_iterations = 100

if __name__== '__main__':
    # # Fit SGT data overall
    # for i, model in enumerate(model_list):
    #     SGT_result = model.fit(SGT_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha', num_exp_restart=100)
    #     SGT_result.to_csv(f'./data/Model/SGT_{model_name_list[i]}.csv', index=False)
    #     IGT_result = model.fit(IGT_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha', num_exp_restart=100)
    #     IGT_result.to_csv(f'./data/Model/IGT_{model_name_list[i]}.csv', index=False)

    # # Now fit the data with moving window
    # for i , model in enumerate(model_list):
    #     SGT_mv = moving_window_model_fitting(SGT_data, model, task='IGT_SGT', window_size=window_size,
    #                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    #                                          num_exp_restart=10, num_training_trials=999)
    #     SGT_mv.to_csv(f'./data/Model/Sliding Window/SGT_{model_name_list[i]}_mv.csv', index=False)
    #     IGT_mv = moving_window_model_fitting(IGT_data, model, task='IGT_SGT', window_size=window_size,
    #                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    #                                          num_exp_restart=10, num_training_trials=999)
    #     IGT_mv.to_csv(f'./data/Model/Sliding Window/IGT_{model_name_list[i]}_mv.csv', index=False)


    # # ==================================================================================================================
    # # Load the model fitting results
    # # ==================================================================================================================
    # load all model fitting results from directory
    model_fitting_results = {}
    model_files = ['SGT_delta.csv', 'SGT_decay.csv', 'SGT_delta_PVL.csv', 'SGT_decay_PVL.csv', 'SGT_delta_asymmetric.csv',
                   'SGT_decay_PVPE.csv', 'IGT_delta.csv', 'IGT_decay.csv', 'IGT_delta_PVL.csv', 'IGT_decay_PVL.csv',
                   'IGT_delta_asymmetric.csv', 'IGT_decay_PVPE.csv']
    param_map = {
        'delta': ['t', 'alpha'],
        'decay': ['t', 'alpha'],
        'delta_PVL': ['t', 'alpha', 'shape', 'la'],
        'decay_PVL': ['t', 'alpha', 'shape', 'la'],
        'delta_asymmetric': ['t', 'alpha_pos', 'alpha_neg'],
        'decay_PVPE': ['t', 'alpha', 'weight', 'shape']
    }

    for file in model_files:
        name = file.replace('.csv', '')
        task_name = name.split('_')[0]
        model_name = '_'.join(name.split('_')[1:])
        model_fitting_results[name] = pd.read_csv(f'./data/Model/{file}')
        model_fitting_results[name].rename(columns={'participant_id': 'Subnum'}, inplace=True)
        model_fitting_results[name] = parameter_extractor(model_fitting_results[name], param_name=param_map[model_name])
        model_fitting_results[name]['Task'] = task_name
        # remove unnecessary columns
        cols_to_keep = ['Subnum', 'AIC', 'BIC'] + param_map[model_name] + ['Task']
        model_fitting_results[name] = model_fitting_results[name][cols_to_keep]
        print(f'Task: {task_name}; Model: {model_name}; Mean BIC: {model_fitting_results[name]["BIC"].mean()}')
        globals()[name] = model_fitting_results[name]

    selected_model = 'decay_PVL'
    selected_model_results = pd.concat([
        globals()[f'SGT_{selected_model}'],
        globals()[f'IGT_{selected_model}']
    ], ignore_index=True)
    dm_summary_modeled = dm_summary.merge(selected_model_results, on=['Subnum', 'Task'], how='left')
    for param in param_map[selected_model]:
        dm_summary_modeled[f'{param}_z'] = dm_summary_modeled.groupby(['Task'])[param].transform(lambda x: (x - x.mean()) / x.std())
    dm_summary_modeled.to_csv('./data/dm_summary_modeled.csv', index=False)

    # Now pivot the modeled summary to wide format
    dm_summary_modeled_wide = dm_summary_modeled.pivot_table(index=['Subnum', 'Condition', 'Order'], columns=['Task'],
                                                              values=['AIC', 'BIC', 't', 'alpha', 'shape', 'la', 't_z',
                                                                      'alpha_z', 'shape_z', 'la_z']).reset_index()
    dm_summary_modeled_wide.columns = ['_'.join(map(str, col)).strip() if col[1] else col[0] for col in dm_summary_modeled_wide.columns.values]
    for param in param_map[selected_model]:
        dm_summary_modeled_wide[f'{param}_Diff'] = dm_summary_modeled_wide[f'{param}_SGT'] - dm_summary_modeled_wide[f'{param}_IGT']
        dm_summary_modeled_wide[f'{param}_Diff_z'] = dm_summary_modeled_wide[f'{param}_Diff'].transform(lambda x: (x - x.mean()) / x.std())
    dm_summary_modeled_wide.to_csv('./data/dm_summary_modeled_wide.csv', index=False)



    avg_rating = pd.read_csv('./data/avg_rating.csv')

    # Load the moving window model fitting results
    SGT_delta_mv = pd.read_csv('./data/Model/Sliding Window/SGT_delta_mv.csv')
    SGT_decay_mv = pd.read_csv('./data/Model/Sliding Window/SGT_decay_mv.csv')
    SGT_decayPVL_mv = pd.read_csv('./data/Model/Sliding Window/SGT_decay_PVL_mv.csv')
    IGT_delta_mv = pd.read_csv('./data/Model/Sliding Window/IGT_delta_mv.csv')
    IGT_decay_mv = pd.read_csv('./data/Model/Sliding Window/IGT_decay_mv.csv')
    IGT_decayPVL_mv = pd.read_csv('./data/Model/Sliding Window/IGT_decay_PVL_mv.csv')


    # Add the condition column
    condition_map = dm_data[['Subnum', 'Condition']].drop_duplicates().set_index('Subnum')['Condition']

    for i, df in enumerate([SGT_delta_mv, SGT_decay_mv, SGT_decayPVL_mv, IGT_delta_mv, IGT_decay_mv, IGT_decayPVL_mv]):
        df['Subnum'] = df['participant_id']
        df['Condition'] = df['Subnum'].map(condition_map)
        # extract parameters
        if i in [0, 1, 3, 4]:  # delta and decay models
            df = parameter_extractor(df, param_name=['t', 'alpha'])
        else:  # decay_PVL model
            df = parameter_extractor(df, param_name=['t', 'alpha', 'shape', 'la'])


    # # Extract best fitting parameters
    # for i, df in enumerate([SGT_dual, SGT_dual_mv, IGT_dual, IGT_dual_mv]):
    #     df = parameter_extractor(df, param_name=['t', 'alpha', 'subj_weight', 't2'])
    #     df['t_diff'] = df['t'] - df['t'].shift(1)
    #     df['alpha_diff'] = df['alpha'] - df['alpha'].shift(1)
    #     df['subj_weight_diff'] = df['subj_weight'] - df['subj_weight'].shift(1)
    #     df['t2_diff'] = df['t2'] - df['t2'].shift(1)

    # Change the window number (This should be changed when counterbalance is used)
    for i, df in enumerate([IGT_delta_mv, IGT_decay_mv, IGT_decayPVL_mv]):
        df['task_id'] = 1

    for i, df in enumerate([SGT_delta_mv, SGT_decay_mv, SGT_decayPVL_mv]):
        df['task_id'] = 2
        df['window_id'] = df['window_id'] + 91

    # Combine the dataframes
    delta_results = pd.concat([IGT_delta_mv, SGT_delta_mv], ignore_index=True)
    decay_results = pd.concat([IGT_decay_mv, SGT_decay_mv], ignore_index=True)
    decayPVL_results = pd.concat([IGT_decayPVL_mv, SGT_decayPVL_mv], ignore_index=True)

    # Add the avg rating to the results
    delta_results = delta_results.merge(avg_rating, on=['Subnum', 'Condition'], how='left')
    decay_results = decay_results.merge(avg_rating, on=['Subnum', 'Condition'], how='left')
    dual_results = dual_results.merge(avg_rating, on=['Subnum', 'Condition'], how='left')

    # Save the results
    delta_results.to_csv('./data/Model/Sliding Window/Delta_Results.csv', index=False)
    decay_results.to_csv('./data/Model/Sliding Window/Decay_Results.csv', index=False)
    dual_results.to_csv('./data/Model/Sliding Window/Dual_Results.csv', index=False)

    # Print the results
    print(f'SGT Delta AIC: {SGT_delta["AIC"].mean()}; SGT Delta BIC: {SGT_delta["BIC"].mean()}')
    print(f'SGT Decay AIC: {SGT_decay["AIC"].mean()}; SGT Decay BIC: {SGT_decay["BIC"].mean()}')
    print(f'SGT Dual AIC: {SGT_dual["AIC"].mean()}; SGT Dual BIC: {SGT_dual["BIC"].mean()}')
    print(f'IGT Delta AIC: {IGT_delta["AIC"].mean()}; IGT Delta BIC: {IGT_delta["BIC"].mean()}')
    print(f'IGT Decay AIC: {IGT_decay["AIC"].mean()}; IGT Decay BIC: {IGT_decay["BIC"].mean()}')
    print(f'IGT Dual AIC: {IGT_dual["AIC"].mean()}; IGT Dual BIC: {IGT_dual["BIC"].mean()}')
    #
    # # ==================================================================================================================
    # # Statistical analysis
    # # ==================================================================================================================
    # # Compare the 91st window with the 92nd window
    # model = delta_results
    # window_91 = model[model['window_id'] == 91]
    # window_92 = model[model['window_id'] == 92]
    #
    # # Perform basic ANOVA
    # print(f'[t-difference] between 91st and 92nd window:')
    # print(f'mean: {window_92.groupby("Condition")["t_diff"].mean()}')
    # print(pg.anova(data=window_92, dv='t_diff', between=['Condition']))
    # # Post-hoc pairwise t-tests for t_diff
    # print("\nPairwise t-tests for t_diff:")
    # t_diff_pairwise = pg.pairwise_tests(data=window_92, dv='t_diff', between='Condition',
    #                         padjust='bonf')
    # print(t_diff_pairwise)
    #
    # print(f'[alpha-difference] between 91st and 92nd window:')
    # print(f'mean: {window_92.groupby("Condition")["alpha_diff"].mean()}')
    # print(pg.anova(data=window_92, dv='alpha_diff', between=['Condition']))
    # # Post-hoc pairwise t-tests for alpha_diff
    # print("\nPairwise t-tests for alpha_diff:")
    # alpha_diff_pairwise = pg.pairwise_tests(data=window_92, dv='alpha_diff', between='Condition',
    #                         padjust='bonf')
    # print(alpha_diff_pairwise)
    #
    # # ==================================================================================================================
    # # Change-Point Detection
    # # ==================================================================================================================
    # # Perform change-point detection on the model fitting results
    # condition_of_interest = 'Nature'
    # avg_alpha = delta_results.groupby(['Condition', 'window_id']).agg({'alpha': 'mean'}).reset_index()
    # avg_alpha = avg_alpha[avg_alpha['Condition'] == condition_of_interest]['alpha'].values
    # algo = rpt.Pelt(model="rbf").fit(avg_alpha)
    # result = algo.predict(pen=3)
    # rpt.display(avg_alpha, result, figsize=(10, 6))
    # plt.title('Change-Point Detection on t parameter')
    # plt.xlabel('Window Number')
    # plt.ylabel('t parameter')
    # plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
    # plt.legend()
    # plt.savefig('./figures/ChangePointDetection.png', dpi=600)
    # plt.show()
    #
    #
    # def detect_rebound_features(task1_alpha, task2_alpha, pen=1):
    #     """
    #     Detects rebound onset, offset, duration, and amplitude relative to Task 1 baseline.
    #
    #     Args:
    #         task1_alpha (float): Baseline alpha (last trial of Task 1).
    #         task2_alpha (np.array): Alpha values from Task 2.
    #         pen (float): Penalty for change-point sensitivity (adjustable).
    #
    #     Returns:
    #         dict: onset, offset, duration, amplitude.
    #     """
    #     series = np.concatenate([[task1_alpha], task2_alpha])
    #     algo = rpt.Pelt(model="rbf").fit(series)
    #     breakpoints = algo.predict(pen=pen)
    #
    #     # Initialize defaults
    #     onset = offset = duration = amplitude = None
    #
    #     # Check that at least two breakpoints exist (onset and offset)
    #     if len(breakpoints) >= 2:
    #         onset = breakpoints[0] - 1  # trial index in Task 2
    #         offset = breakpoints[1] - 1
    #
    #         duration = offset - onset
    #
    #         # Compute amplitude as difference in mean alpha between segments
    #         baseline_mean = series[:onset + 1].mean()
    #         rebound_mean = series[onset + 1:offset + 1].mean()
    #         amplitude = rebound_mean - baseline_mean
    #
    #     return {
    #         "onset": onset,
    #         "offset": offset,
    #         "duration": duration,
    #         "amplitude": amplitude
    #     }
    #
    #
    # def detect_group_features(df, pen=1):
    #     baseline_alpha = df[df['task_id'] == 1].sort_values('window_id')['alpha'].iloc[-1]
    #     task2_alpha = df[df['task_id'] == 2].sort_values('window_id')['alpha'].values
    #     return pd.Series(detect_rebound_features(baseline_alpha, task2_alpha, pen=pen))
    #
    #
    # # Apply per participant and condition
    # results = delta_results.groupby(['Subnum', 'Condition']).apply(detect_group_features).reset_index()
    #
    # print(results.head())
    #
    # ==================================================================================================================
    # Plot the model fitting results
    # ==================================================================================================================
    # Preprocess the data for plotting
    models = ['Delta', 'Decay', 'Decay_PVL']
    for i, df in enumerate([delta_results, decay_results, decayPVL_results]):
        df['Condition'] = pd.Categorical(df['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)


    # Draw the model fitting results for all models and parameters
    for i, model_result in enumerate([delta_results, decay_results, decayPVL_results]):
        for param in ['t', 'alpha', 'shape', 'la']:
            # check if the parameter exists in the model result
            if param not in model_result.columns:
                continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=model_result, x='window_id', y=param, hue='Condition', errorbar='se')
            plt.xlabel('Window Number')
            plt.ylabel(param)
            plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
            plt.savefig(f'./figures/{param}ByWindow_{models[i]}.png', dpi=600)
            plt.show()

    # # # Draw BIC for all models
    # # delta_results['Model'] = 'Delta'
    # # decay_results['Model'] = 'Decay'
    # # dual_results['Model'] = 'Dual'
    # # all_results = pd.concat([delta_results, decay_results], ignore_index=True)
    # # # plot three figures for three conditions
    # # for condition in all_results['Condition'].unique():
    # #     condition_data = all_results[all_results['Condition'] == condition]
    # #     plt.figure(figsize=(10, 6))
    # #     sns.lineplot(data=condition_data, x='window_id', y='BIC', hue='Model', errorbar='se')
    # #     plt.xlabel('Window Number')
    # #     plt.ylabel('BIC')
    # #     plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
    # #     plt.title(f'BIC by Window for {condition} Condition')
    # #     plt.savefig(f'./figures/BICByWindow_{condition}.png', dpi=600)
    # #     plt.show()



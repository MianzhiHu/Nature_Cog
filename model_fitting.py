import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import ruptures as rpt
from utils.ComputationalModeling import (ComputationalModels, dict_generator, moving_window_model_fitting,
                                         parameter_extractor)
from utils.DualProcess import DualProcessModel

# ======================================================================================================================
# Load the data
# ======================================================================================================================
dm_data = pd.read_csv('./data/dm_data.csv')
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
delta = ComputationalModels('delta', num_trials=100, condition='Both', task='IGT_SGT')
decay = ComputationalModels('decay', num_trials=100, condition='Both', task='IGT_SGT')
dual_process = DualProcessModel(num_trials=100, task='IGT_SGT', default_EV=0.0)

# Set window parameters
window_size = 10
n_iterations = 100

if __name__== '__main__':
    # # Fit SGT data overall
    # SGT_delta = delta.fit(SGT_dict, num_iterations=n_iterations)
    # SGT_decay = decay.fit(SGT_dict, num_iterations=n_iterations)
    # SGT_dual = dual_process.fit(SGT_dict, num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                             arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=2)
    #
    # # Save the results
    # SGT_delta.to_csv('./data/Model/SGT_delta.csv', index=False)
    # SGT_decay.to_csv('./data/Model/SGT_decay.csv', index=False)
    # SGT_dual.to_csv('./data/Model/SGT_dual.csv', index=False)
    #
    # # Fit IGT data overall
    # IGT_delta = delta.fit(IGT_dict, num_iterations=n_iterations)
    # IGT_decay = decay.fit(IGT_dict, num_iterations=n_iterations)
    # IGT_dual = dual_process.fit(IGT_dict, num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                             arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=2)
    #
    # # Save the results
    # IGT_delta.to_csv('./data/Model/IGT_delta.csv', index=False)
    # IGT_decay.to_csv('./data/Model/IGT_decay.csv', index=False)
    # IGT_dual.to_csv('./data/Model/IGT_dual.csv', index=False)
    #
    # # Now fit the data with moving window
    # # Fit SGT
    # SGT_delta_mv = moving_window_model_fitting(SGT_data, delta, task='IGT_SGT', window_size=window_size,
    #                                            num_iterations=n_iterations)
    # SGT_decay_mv = moving_window_model_fitting(SGT_data, decay, task='IGT_SGT', window_size=window_size,
    #                                            num_iterations=n_iterations)
    # SGT_dual_mv = moving_window_model_fitting(SGT_data, dual_process, task='IGT_SGT', window_size=window_size,
    #                                           num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                           arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency',
    #                                           num_t=2)
    #
    # # Save the results
    # SGT_delta_mv.to_csv('./data/Model/Sliding Window/SGT_delta_mv.csv', index=False)
    # SGT_decay_mv.to_csv('./data/Model/Sliding Window/SGT_decay_mv.csv', index=False)
    # SGT_dual_mv.to_csv('./data/Model/Sliding Window/SGT_dual_mv.csv', index=False)
    #
    # # Fit IGT
    # IGT_delta_mv = moving_window_model_fitting(IGT_data, delta, task='IGT_SGT', window_size=window_size,
    #                                            num_iterations=n_iterations)
    # IGT_decay_mv = moving_window_model_fitting(IGT_data, decay, task='IGT_SGT', window_size=window_size,
    #                                            num_iterations=n_iterations)
    # IGT_dual_mv = moving_window_model_fitting(IGT_data, dual_process, task='IGT_SGT', window_size=window_size,
    #                                           num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                           arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency',
    #                                           num_t=2)
    #
    # # Save the results
    # IGT_delta_mv.to_csv('./data/Model/Sliding Window/IGT_delta_mv.csv', index=False)
    # IGT_decay_mv.to_csv('./data/Model/Sliding Window/IGT_decay_mv.csv', index=False)
    # IGT_dual_mv.to_csv('./data/Model/Sliding Window/IGT_dual_mv.csv', index=False)

    # ==================================================================================================================
    # Load the model fitting results
    # ==================================================================================================================
    SGT_delta = pd.read_csv('./data/Model/SGT_delta.csv')
    SGT_decay = pd.read_csv('./data/Model/SGT_decay.csv')
    SGT_dual = pd.read_csv('./data/Model/SGT_dual.csv')
    IGT_delta = pd.read_csv('./data/Model/IGT_delta.csv')
    IGT_decay = pd.read_csv('./data/Model/IGT_decay.csv')
    IGT_dual = pd.read_csv('./data/Model/IGT_dual.csv')
    SGT_delta_mv = pd.read_csv('./data/Model/Sliding Window/SGT_delta_mv.csv')
    SGT_decay_mv = pd.read_csv('./data/Model/Sliding Window/SGT_decay_mv.csv')
    SGT_dual_mv = pd.read_csv('./data/Model/Sliding Window/SGT_dual_mv.csv')
    IGT_delta_mv = pd.read_csv('./data/Model/Sliding Window/IGT_delta_mv.csv')
    IGT_decay_mv = pd.read_csv('./data/Model/Sliding Window/IGT_decay_mv.csv')
    IGT_dual_mv = pd.read_csv('./data/Model/Sliding Window/IGT_dual_mv.csv')

    avg_rating = pd.read_csv('./data/avg_rating.csv')

    # Change the subject number column name to Subnum
    for df in [SGT_delta, SGT_decay, SGT_dual, IGT_delta, IGT_decay, IGT_dual,
              SGT_delta_mv, SGT_decay_mv, SGT_dual_mv, IGT_delta_mv, IGT_decay_mv, IGT_dual_mv]:
        df.rename(columns={'participant_id': 'Subnum'}, inplace=True)

    # Add the condition column
    condition_map = dm_data[['Subnum', 'Condition']].drop_duplicates().set_index('Subnum')['Condition']

    for df in [SGT_delta, SGT_decay, SGT_dual, IGT_delta, IGT_decay, IGT_dual]:
        df['Condition'] = df['Subnum'].map(condition_map)

    for df in [SGT_delta_mv, SGT_decay_mv, SGT_dual_mv, IGT_delta_mv, IGT_decay_mv, IGT_dual_mv]:
        df['Condition'] = df['Subnum'].map(condition_map)

    # Extract best fitting parameters
    for i, df in enumerate([SGT_delta, SGT_decay, IGT_delta, IGT_decay,
                            SGT_delta_mv, SGT_decay_mv, IGT_delta_mv, IGT_decay_mv]):
        df = parameter_extractor(df, param_name=['t', 'alpha'])
        df['t_diff'] = df['t'] - df['t'].shift(1)
        df['alpha_diff'] = df['alpha'] - df['alpha'].shift(1)

    for i, df in enumerate([SGT_dual, SGT_dual_mv, IGT_dual, IGT_dual_mv]):
        df = parameter_extractor(df, param_name=['t', 'alpha', 'subj_weight', 't2'])
        df['t_diff'] = df['t'] - df['t'].shift(1)
        df['alpha_diff'] = df['alpha'] - df['alpha'].shift(1)
        df['subj_weight_diff'] = df['subj_weight'] - df['subj_weight'].shift(1)
        df['t2_diff'] = df['t2'] - df['t2'].shift(1)

    # Change the window number (This should be changed when counterbalance is used)
    for i, df in enumerate([IGT_delta_mv, IGT_decay_mv, IGT_dual_mv]):
        df['window_id'] = df['window_id'] + 91
        df['task_id'] = 2

    for i, df in enumerate([SGT_delta_mv, SGT_decay_mv, SGT_dual_mv]):
        df['task_id'] = 1

    # Combine the dataframes
    delta_results = pd.concat([SGT_delta_mv, IGT_delta_mv], ignore_index=True)
    decay_results = pd.concat([SGT_decay_mv, IGT_decay_mv], ignore_index=True)
    dual_results = pd.concat([SGT_dual_mv, IGT_dual_mv], ignore_index=True)

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

    # ==================================================================================================================
    # Statistical analysis
    # ==================================================================================================================
    # Compare the 91st window with the 92nd window
    model = delta_results
    window_91 = model[model['window_id'] == 91]
    window_92 = model[model['window_id'] == 92]

    # Perform basic ANOVA
    print(f'[t-difference] between 91st and 92nd window:')
    print(f'mean: {window_92.groupby("Condition")["t_diff"].mean()}')
    print(pg.anova(data=window_92, dv='t_diff', between=['Condition']))
    # Post-hoc pairwise t-tests for t_diff
    print("\nPairwise t-tests for t_diff:")
    t_diff_pairwise = pg.pairwise_tests(data=window_92, dv='t_diff', between='Condition',
                            padjust='bonf')
    print(t_diff_pairwise)

    print(f'[alpha-difference] between 91st and 92nd window:')
    print(f'mean: {window_92.groupby("Condition")["alpha_diff"].mean()}')
    print(pg.anova(data=window_92, dv='alpha_diff', between=['Condition']))
    # Post-hoc pairwise t-tests for alpha_diff
    print("\nPairwise t-tests for alpha_diff:")
    alpha_diff_pairwise = pg.pairwise_tests(data=window_92, dv='alpha_diff', between='Condition',
                            padjust='bonf')
    print(alpha_diff_pairwise)

    # ==================================================================================================================
    # Change-Point Detection
    # ==================================================================================================================
    # Perform change-point detection on the model fitting results
    condition_of_interest = 'Nature'
    avg_alpha = delta_results.groupby(['Condition', 'window_id']).agg({'alpha': 'mean'}).reset_index()
    avg_alpha = avg_alpha[avg_alpha['Condition'] == condition_of_interest]['alpha'].values
    algo = rpt.Pelt(model="rbf").fit(avg_alpha)
    result = algo.predict(pen=3)
    rpt.display(avg_alpha, result, figsize=(10, 6))
    plt.title('Change-Point Detection on t parameter')
    plt.xlabel('Window Number')
    plt.ylabel('t parameter')
    plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
    plt.legend()
    plt.savefig('./figures/ChangePointDetection.png', dpi=600)
    plt.show()


    def detect_rebound_features(task1_alpha, task2_alpha, pen=1):
        """
        Detects rebound onset, offset, duration, and amplitude relative to Task 1 baseline.

        Args:
            task1_alpha (float): Baseline alpha (last trial of Task 1).
            task2_alpha (np.array): Alpha values from Task 2.
            pen (float): Penalty for change-point sensitivity (adjustable).

        Returns:
            dict: onset, offset, duration, amplitude.
        """
        series = np.concatenate([[task1_alpha], task2_alpha])
        algo = rpt.Pelt(model="rbf").fit(series)
        breakpoints = algo.predict(pen=pen)

        # Initialize defaults
        onset = offset = duration = amplitude = None

        # Check that at least two breakpoints exist (onset and offset)
        if len(breakpoints) >= 2:
            onset = breakpoints[0] - 1  # trial index in Task 2
            offset = breakpoints[1] - 1

            duration = offset - onset

            # Compute amplitude as difference in mean alpha between segments
            baseline_mean = series[:onset + 1].mean()
            rebound_mean = series[onset + 1:offset + 1].mean()
            amplitude = rebound_mean - baseline_mean

        return {
            "onset": onset,
            "offset": offset,
            "duration": duration,
            "amplitude": amplitude
        }


    def detect_group_features(df, pen=1):
        baseline_alpha = df[df['task_id'] == 1].sort_values('window_id')['alpha'].iloc[-1]
        task2_alpha = df[df['task_id'] == 2].sort_values('window_id')['alpha'].values
        return pd.Series(detect_rebound_features(baseline_alpha, task2_alpha, pen=pen))


    # Apply per participant and condition
    results = delta_results.groupby(['Subnum', 'Condition']).apply(detect_group_features).reset_index()

    print(results.head())

    # # ==================================================================================================================
    # # Plot the model fitting results
    # # ==================================================================================================================
    # # Preprocess the data for plotting
    # models = ['Delta', 'Decay', 'Dual']
    # for i, df in enumerate([delta_results, decay_results, dual_results]):
    #     df['Condition'] = pd.Categorical(df['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
    #
    #
    # # Draw the model fitting results for all models and parameters
    # for i, model_result in enumerate([delta_results, decay_results, dual_results]):
    #     for param in ['t', 'alpha', 'subj_weight', 't2']:
    #         # check if the parameter exists in the model result
    #         if param not in model_result.columns:
    #             continue
    #         plt.figure(figsize=(10, 6))
    #         sns.lineplot(data=model_result, x='window_id', y=param, hue='Condition', errorbar='se')
    #         plt.xlabel('Window Number')
    #         plt.ylabel(param)
    #         plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
    #         plt.savefig(f'./figures/{param}ByWindow_{models[i]}.png', dpi=600)
    #         plt.show()
    #
    # # Draw BIC for all models
    # delta_results['Model'] = 'Delta'
    # decay_results['Model'] = 'Decay'
    # dual_results['Model'] = 'Dual'
    # all_results = pd.concat([delta_results, decay_results], ignore_index=True)
    # # plot three figures for three conditions
    # for condition in all_results['Condition'].unique():
    #     condition_data = all_results[all_results['Condition'] == condition]
    #     plt.figure(figsize=(10, 6))
    #     sns.lineplot(data=condition_data, x='window_id', y='BIC', hue='Model', errorbar='se')
    #     plt.xlabel('Window Number')
    #     plt.ylabel('BIC')
    #     plt.axvline(x=91, color='red', linestyle='--', label='Task Switch')
    #     plt.title(f'BIC by Window for {condition} Condition')
    #     plt.savefig(f'./figures/BICByWindow_{condition}.png', dpi=600)
    #     plt.show()



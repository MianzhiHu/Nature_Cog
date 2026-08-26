import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from utils.ComputationalModeling import (BIC_weights, ComputationalModels, bayes_factor,
                                         behavioral_moving_window, compute_exceedance_prob, dict_generator,
                                         moving_window_model_fitting, parameter_extractor, parse_numeric_history,
                                         vb_model_selection)
import functools
import ast
from matplotlib import font_manager as fm


if __name__== '__main__':
    # ======================================================================================================================
    # Load the data
    # ======================================================================================================================
    E1_dm_data = pd.read_csv('./data/E1_dm_data.csv')
    E1_dm_summary = pd.read_csv('./data/E1_dm_summary.csv')
    E1_dm_data['Trial'] = E1_dm_data['Trial'].astype(int)
    E1_dm_data['KeyResponse'] = E1_dm_data['KeyResponse'].astype(int)
    E1_dm_data['KeyResponse'] = E1_dm_data['KeyResponse'] - 1
    E1_img_data = pd.read_csv('./data/E1_img_data.csv')
    E1_taskfirst_data = E1_dm_data[E1_dm_data['Task'] == 1].copy()
    E1_tasksecond_data = E1_dm_data[E1_dm_data['Task'] == 2].copy()

    E2_data = pd.read_csv('./data/E2_all_data.csv')
    E2_data['Trial'] = E2_data['Trial'].astype(int)
    E2_data['KeyResponse'] = E2_data['KeyResponse'].astype(int)
    E2_data['KeyResponse'] = E2_data['KeyResponse'] - 1

    # ======================================================================================================================
    # Data Preprocessing
    # ======================================================================================================================
    E1_taskfirst_dict = dict_generator(E1_taskfirst_data, 'IGT_SGT')
    E1_tasksecond_dict = dict_generator(E1_tasksecond_data, 'IGT_SGT')
    E2_dict = dict_generator(E2_data, 'IGT_SGT')

    # ======================================================================================================================
    # Model Fitting
    # ======================================================================================================================
    # Define the model parameters
    delta = ComputationalModels('delta', task='IGT_SGT')
    decay = ComputationalModels('decay', task='IGT_SGT')
    delta_PVL = ComputationalModels('delta_PVL', task='IGT_SGT')
    delta_PVL_relative = ComputationalModels('delta_PVL_relative', task='IGT_SGT')
    decay_PVL = ComputationalModels('decay_PVL', task='IGT_SGT')
    decay_PVL_relative = ComputationalModels('decay_PVL_relative', task='IGT_SGT')
    delta_asymmetric = ComputationalModels('delta_asymmetric', task='IGT_SGT')
    decay_PVPE = ComputationalModels('decay_PVPE', task='IGT_SGT')
    decay_win = ComputationalModels('decay_win', task='IGT_SGT')
    WSLS_avg = ComputationalModels('WSLS_avg', task='IGT_SGT')
    WSLS_delta = ComputationalModels('WSLS_delta', task='IGT_SGT')
    kalman_filter = ComputationalModels('kalman_filter', task='IGT_SGT')
    kalman_decay = ComputationalModels('kalman_decay', task='IGT_SGT')
    kalman_filter_bonus = ComputationalModels('kalman_filter_bonus', task='IGT_SGT')
    kalman_decay_bonus = ComputationalModels('kalman_decay_bonus', task='IGT_SGT')

    model_list = [delta, decay, delta_PVL, delta_PVL_relative, decay_PVL, decay_PVL_relative, delta_asymmetric,
                  decay_PVPE, decay_win, WSLS_avg, WSLS_delta]
    # model_list = [decay_PVL, decay_PVL_relative, delta_asymmetric]
    # model_list = [kalman_filter, kalman_decay, kalman_filter_bonus, kalman_decay_bonus]
    # model_list = [kalman_decay]
    model_name_list = [name for name, obj in globals().items() if any(obj is m for m in model_list)]

    # Set window parameters
    window_size = 30
    n_iterations = 200

    # # Fit E1 data
    # # When fitting kalman filter models, use initial_EV=[50.0, 50.0, 50.0, 50.0], initial_var=[16.0, 16.0, 16.0, 16.0]
    # for i, model in enumerate(model_list):
    #     taskfirst_path = f'./data/Model/{model_name_list[i]}_1st.csv'
    #     if os.path.exists(taskfirst_path):
    #         print(f'{taskfirst_path} already exists. Skipping.')
    #     else:
    #         taskfirst_result = model.fit(E1_taskfirst_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    #                                      num_exp_restart=150)
    #         taskfirst_result.to_csv(taskfirst_path, index=False)
    #
    #     tasksecond_path = f'./data/Model/{model_name_list[i]}_2nd.csv'
    #     if os.path.exists(tasksecond_path):
    #         print(f'{tasksecond_path} already exists. Skipping.')
    #     else:
    #         tasksecond_result = model.fit(E1_tasksecond_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    #                                       num_exp_restart=150)
    #         tasksecond_result.to_csv(tasksecond_path, index=False)
    #
    # # # Now fit E1 data with moving window
    # # for i , model in enumerate(model_list):
    # #     taskfirst_mv = moving_window_model_fitting(E1_taskfirst_data, model, task='IGT_SGT', window_size=window_size,
    # #                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    # #                                          num_exp_restart=999, num_training_trials=999, initial_EV=[50.0, 50.0, 50.0, 50.0], initial_var=[16.0, 16.0, 16.0, 16.0])
    # #     taskfirst_mv.to_csv(f'./data/Model/Sliding Window/{model_name_list[i]}_1st.csv', index=False)
    # #     tasksecond_mv = moving_window_model_fitting(E1_tasksecond_data, model, task='IGT_SGT', window_size=window_size,
    # #                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    # #                                          num_exp_restart=999, num_training_trials=999, initial_EV=[50.0, 50.0, 50.0, 50.0], initial_var=[16.0, 16.0, 16.0, 16.0])
    # #     tasksecond_mv.to_csv(f'./data/Model/Sliding Window/{model_name_list[i]}_2nd.csv', index=False)
    #
    # # Fit E2 data
    # for i, model in enumerate(model_list):
    #     E2_path = f'./data/Model/{model_name_list[i]}_E2.csv'
    #     if os.path.exists(E2_path):
    #         print(f'{E2_path} already exists. Skipping.')
    #     else:
    #         E2_result = model.fit(E2_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    #                               num_exp_restart=150)
    #         E2_result.to_csv(E2_path, index=False)
    #
    # # # Now fit E2 data with moving window
    # # for i , model in enumerate(model_list):
    # #     E2_mv = moving_window_model_fitting(E2_data, model, task='IGT_SGT', window_size=window_size,
    # #                                          num_iterations=n_iterations, initial_mode='first_trial_no_alpha',
    # #                                          num_exp_restart=999, num_training_trials=999, initial_EV=[50.0, 50.0, 50.0, 50.0], initial_var=[16.0, 16.0, 16.0, 16.0])
    # #     E2_mv.to_csv(f'./data/Model/Sliding Window/{model_name_list[i]}_E2.csv', index=False)

    # ==================================================================================================================
    # Load the model fitting results
    # ==================================================================================================================
    # load all model fitting results from directory
    model_fitting_results = {}
    folder_path = './data/Model/'
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    param_map = {
        'delta': ['t', 'alpha'],
        'decay': ['t', 'alpha'],
        'delta_PVL': ['t', 'alpha', 'shape', 'la'],
        'delta_PVL_relative': ['t', 'alpha', 'shape', 'la'],
        'decay_PVL': ['t', 'alpha', 'shape', 'la'],
        'decay_PVL_relative': ['t', 'alpha', 'shape', 'la'],
        'delta_asymmetric': ['t', 'alpha_pos', 'alpha_neg'],
        'decay_PVPE': ['t', 'alpha', 'weight', 'shape'],
        'decay_win': ['t', 'alpha'],
        'dual_process': ['t', 'alpha', 'subj_weight'],
        'kalman_filter': ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center'],
        'kalman_filter_bonus': ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center', 'exploration_bonus'],
        'kalman_decay': ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center'],
        'kalman_decay_bonus': ['t', 'dis_sd', 'noise_sd', 'decay', 'decay_center', 'exploration_bonus'],
        'kalman_simple': ['t', 'dis_sd'],
        'WSLS_avg': ['p_ws', 'p_ls'],
        'WSLS_delta':['a', 'p_ws', 'p_ls']
    }

    for file in model_files:
        name = file.replace('.csv', '')
        task_name = name.split('_')[-1]
        model_name = '_'.join(name.split('_')[:-1])
        model_fitting_results[name] = pd.read_csv(f'./data/Model/{file}')
        model_fitting_results[name].rename(columns={'participant_id': 'Subnum'}, inplace=True)
        model_fitting_results[name] = parameter_extractor(model_fitting_results[name], param_name=param_map[model_name])
        model_fitting_results[name]['Task'] = task_name
        model_fitting_results[name]['Model'] = model_name
        if task_name == 'E2':
            condition_map = E2_data[['Subnum', 'Condition']].drop_duplicates().set_index('Subnum')['Condition']
        else:
            condition_map = E1_dm_data[['Subnum', 'Condition']].drop_duplicates().set_index('Subnum')['Condition']
        model_fitting_results[name]['Condition'] = model_fitting_results[name]['Subnum'].map(condition_map)
        # remove unnecessary columns
        if model_name == 'dual_process':
            cols_to_keep = (['Subnum', 'AIC', 'BIC', 'gau_exploitation', 'dir_exploitation', 'EV_history'] + param_map[model_name] +
                            ['Task', 'Model', 'Condition'])
        else:
            cols_to_keep = ['Subnum', 'AIC', 'BIC', 'exploitation', 'EV_history'] + param_map[model_name] + ['Task', 'Model', 'Condition']
        model_fitting_results[name] = model_fitting_results[name][cols_to_keep]
        print(f'Task: {task_name}; Model: {model_name}; Mean BIC: {model_fitting_results[name]["BIC"].mean()}')
        globals()[name] = model_fitting_results[name]

    # Create a df from model fitting results containing all models
    all_model_results = pd.concat(model_fitting_results).reset_index(level=0)

    # ------------------------------------------------------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------------------------------------------------------
    comparison_models = [
        'delta', 'delta_PVL_relative', 'delta_asymmetric', 'decay', 'decay_PVL_relative',
        'decay_PVPE', 'decay_win', 'WSLS_avg', 'WSLS_delta', 'kalman_filter',
        'kalman_decay', 'kalman_filter_bonus', 'kalman_decay_bonus'
    ]
    comparison_sets = {'E1 Task 1': '1st', 'E1 Task 2': '2nd', 'E2': 'E2'}
    comparison_results = []
    np.random.seed(20260813)

    for comparison_set, task_name in comparison_sets.items():
        task_results = all_model_results[
            (all_model_results['Task'] == task_name)
            & (all_model_results['Model'].isin(comparison_models))
        ].copy()
        bic_wide = task_results.pivot(index='Subnum', columns='Model', values='BIC').reindex(
            columns=comparison_models
        )
        if bic_wide.isna().any().any():
            raise ValueError(f'{comparison_set} does not have complete BIC results for all 13 models.')

        mean_bic = bic_wide.mean()
        best_model = mean_bic.idxmin()
        best_model_results = bic_wide[[best_model]].rename(columns={best_model: 'BIC'})
        n_best_fit = bic_wide.idxmin(axis=1).value_counts()
        bic_weight = BIC_weights(mean_bic.to_numpy())

        log_evidences = bic_wide.to_numpy() / (-2)
        alpha, g = vb_model_selection(log_evidences, tol=1e-12, max_iter=50000)
        model_frequency = alpha / np.sum(alpha)
        exceedance_probability = compute_exceedance_prob(alpha, n_samples=100000)

        for model_index, model in enumerate(comparison_models):
            current_model_results = bic_wide[[model]].rename(columns={model: 'BIC'})
            comparison_results.append({
                'Comparison Set': comparison_set,
                'Model': model,
                'N Participants': len(bic_wide),
                'Mean BIC': mean_bic[model],
                'N Best Fit': n_best_fit.get(model, 0),
                'BIC Weight': bic_weight[model_index],
                'Bayes Factor': bayes_factor(current_model_results, best_model_results),
                'VBMS Alpha': alpha[model_index],
                'VBMS Model Frequency': model_frequency[model_index],
                'VBMS Exceedance Probability': exceedance_probability[model_index],
                'Best Model': best_model
            })

    model_comparison_results = pd.DataFrame(comparison_results)
    model_comparison_results['N Best Fit'] = model_comparison_results['N Best Fit'].astype(int)
    model_comparison_results.to_csv('./data/model_comparison_results.csv', index=False)

    # Exploration all models
    all_model_results['EV_rank'] = all_model_results['exploitation'].apply(ast.literal_eval)
    all_model_results['EV_rank'] = all_model_results['EV_rank'].apply(
        lambda ranks: [int(str(rank)[0]) for rank in ranks])
    all_model_results['exploration'] = all_model_results['EV_rank'].apply(
        lambda ranks: ['exploitation' if rank == 1 else 'exploration' for rank in ranks])
    all_model_results['EV_history'] = all_model_results['EV_history'].apply(parse_numeric_history)
    df_exploded = all_model_results.explode(['exploration', 'EV_rank', 'EV_history'])

    # ------------------------------------------------------------------------------------------------------------------
    # E1
    # ------------------------------------------------------------------------------------------------------------------
    E1_df_exploded = df_exploded[df_exploded['Task'] != 'E2'].copy()
    E1_df_exploded['EV_history'] = pd.to_numeric(E1_df_exploded['EV_history'])
    E1_df_exploded['EV_rank'] = pd.to_numeric(E1_df_exploded['EV_rank'])
    E1_df_exploded['rank_2'] = (E1_df_exploded['EV_rank'] == 2).astype(int)
    E1_exploration_summary = (E1_df_exploded.groupby(['Subnum', 'Model', 'Condition', 'Task'])['exploration'].
                              value_counts().unstack(fill_value=0).reset_index())
    E1_exploration_summary['Exploration_Rate'] = E1_exploration_summary['exploration'] / 149
    E1_exploration_summary['Exploration_Rate_z'] = (E1_exploration_summary.groupby(['Model'])['Exploration_Rate'].
                                                    transform(lambda x: (x - x.mean()) / x.std()))
    E1_EV_summary = E1_df_exploded.groupby(['Subnum', 'Model', 'Condition', 'Task']).agg(
        EV_history=('EV_history', 'mean'),
        EV_rank=('EV_rank', 'mean'),
        rank_2=('rank_2', 'mean')
    ).reset_index()
    E1_exploration_only_summary = (
        E1_df_exploded[E1_df_exploded['exploration'] == 'exploration']
        .groupby(['Subnum', 'Model', 'Condition', 'Task'])
        .agg(
            EV_history_exploration=('EV_history', 'mean'),
            rank_2_exploration_rate=('rank_2', 'mean')
        )
        .reset_index()
    )

    # Calculate the difference in exploration rate by task
    E1_exploration_wide = E1_exploration_summary.pivot_table(
        index=['Subnum', 'Model', 'Condition'],
        columns='Task',
        values='Exploration_Rate'
    ).reset_index()
    E1_exploration_wide.columns.name = None
    E1_exploration_wide['Exploration_Diff'] = E1_exploration_wide['2nd'] - E1_exploration_wide['1st']
    E1_exploration_wide['Exploration_Diff_z'] = E1_exploration_wide.groupby(['Model'])['Exploration_Diff'].transform(lambda x: (x - x.mean()) / x.std())


    # # plot the mean BIC for each model and task and condition
    # plt.figure(figsize=(12, 6))
    # sns.catplot(data=E1_exploration_summary, x='Condition', y='Exploration_Rate_z', hue='Task', col='Model', kind='bar', height=6, aspect=1, errorbar='ci')
    # plt.savefig('./figures/Mean_BIC_by_Model_Task_Condition.png', dpi=600)
    # plt.show()
    #
    # # plot the mean BIC for each model and task and condition
    # plt.figure(figsize=(12, 6))
    # sns.catplot(data=E1_exploration_wide, x='Condition', y='Exploration_Diff', col='Model', kind='bar', height=6, aspect=1)
    # plt.savefig('./figures/Exploration_Diff_by_Model_Task_Condition.png', dpi=600)
    # plt.show()


    selected_model = 'kalman_decay'
    selected_model_results = pd.concat([
        globals()[f'{selected_model}_1st'],
        globals()[f'{selected_model}_2nd']
    ], ignore_index=True)
    selected_model_results = selected_model_results.drop(columns=['exploitation', 'EV_history'], errors='ignore')
    dm_summary_modeled = pd.concat([selected_model_results], ignore_index=True)
    dm_summary_modeled = pd.merge(dm_summary_modeled, E1_exploration_summary[E1_exploration_summary['Model'] == selected_model],
                                  on=['Subnum', 'Condition', 'Task', 'Model'], how='left')
    dm_summary_modeled = pd.merge(dm_summary_modeled, E1_EV_summary[E1_EV_summary['Model'] == selected_model],
                                  on=['Subnum', 'Condition', 'Task', 'Model'], how='left')
    dm_summary_modeled = pd.merge(
        dm_summary_modeled,
        E1_exploration_only_summary[E1_exploration_only_summary['Model'] == selected_model],
        on=['Subnum', 'Condition', 'Task', 'Model'],
        how='left')

    # Rename task with 1st being 1 as integer and 2nd being 2 as integer too
    task_mapping = {'1st': 1, '2nd': 2}
    dm_summary_modeled['Task'] = dm_summary_modeled['Task'].map(task_mapping)
    dm_summary_modeled = pd.merge(dm_summary_modeled, E1_dm_summary,
                                  on=['Subnum', 'Condition', 'Task'], how='left')
    for param in param_map[selected_model]:
        dm_summary_modeled[f'{param}_z'] = dm_summary_modeled.groupby(['Task'])[param].transform(lambda x: (x - x.mean()) / x.std())
    dm_summary_modeled.to_csv('./data/dm_summary_modeled.csv', index=False)

    E1_exploration = df_exploded[df_exploded['Model'] == selected_model]
    E1_exploration = E1_exploration[E1_exploration['level_0'] != 'kalman_decay_E2']
    E1_exploration['Task'] = E1_exploration['Task'].map(task_mapping)
    E1_exploration['rank_2'] = (pd.to_numeric(E1_exploration['EV_rank']) == 2).astype(int)
    E1_exploration.to_csv('./data/E1_exploration_data.csv', index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # E2
    # ------------------------------------------------------------------------------------------------------------------
    E2_df_exploded = df_exploded[df_exploded['Task'] == 'E2'].copy()
    E2_df_exploded['EV_history'] = pd.to_numeric(E2_df_exploded['EV_history'])
    E2_df_exploded['EV_rank'] = pd.to_numeric(E2_df_exploded['EV_rank'])
    E2_df_exploded['rank_2'] = (E2_df_exploded['EV_rank'] == 2).astype(int)
    E2_exploration_summary = (E2_df_exploded.groupby(['Subnum', 'Model', 'Condition', 'Task'])['exploration'].
                              value_counts().unstack(fill_value=0).reset_index())
    E2_exploration_summary['Exploration_Rate'] = E2_exploration_summary['exploration'] / 249
    E2_exploration_summary['Exploration_Rate_z'] = (E2_exploration_summary.groupby(['Model'])['Exploration_Rate'].
                                                    transform(lambda x: (x - x.mean()) / x.std()))
    E2_EV_summary = E2_df_exploded.groupby(['Subnum', 'Model', 'Condition', 'Task']).agg(
        EV_history=('EV_history', 'mean'),
        EV_rank=('EV_rank', 'mean'),
        rank_2=('rank_2', 'mean')
    ).reset_index()
    E2_exploration_only_summary = (
        E2_df_exploded[E2_df_exploded['exploration'] == 'exploration']
        .groupby(['Subnum', 'Model', 'Condition', 'Task'])
        .agg(
            EV_history_exploration=('EV_history', 'mean'),
            rank_2_exploration_rate=('rank_2', 'mean')
        )
        .reset_index()
    )

    E2_selected_model_results = globals()[f'{selected_model}_E2'].drop(
        columns=['exploitation', 'EV_history'], errors='ignore')
    E2_dm_summary_modeled = pd.merge(
        E2_selected_model_results,
        E2_exploration_summary[E2_exploration_summary['Model'] == selected_model],
        on=['Subnum', 'Condition', 'Task', 'Model'],
        how='left')
    E2_dm_summary_modeled = pd.merge(
        E2_dm_summary_modeled,
        E2_EV_summary[E2_EV_summary['Model'] == selected_model],
        on=['Subnum', 'Condition', 'Task', 'Model'],
        how='left')
    E2_dm_summary_modeled = pd.merge(
        E2_dm_summary_modeled,
        E2_exploration_only_summary[E2_exploration_only_summary['Model'] == selected_model],
        on=['Subnum', 'Condition', 'Task', 'Model'],
        how='left')
    for param in param_map[selected_model]:
        E2_dm_summary_modeled[f'{param}_z'] = (
            E2_dm_summary_modeled[param] - E2_dm_summary_modeled[param].mean()
        ) / E2_dm_summary_modeled[param].std()
    E2_dm_summary_modeled.to_csv('./data/E2_dm_summary_modeled.csv', index=False)

    E2_exploration = df_exploded[(df_exploded['Model'] == selected_model) & (df_exploded['Task'] == 'E2')]
    E2_exploration['rank_2'] = (pd.to_numeric(E2_exploration['EV_rank']) == 2).astype(int)
    E2_exploration.to_csv('./data/E2_exploration_data.csv', index=False)
    #
    # ------------------------------------------------------------------------------------------------------------------
    # E2
    # ------------------------------------------------------------------------------------------------------------------
    E2_df_exploded = df_exploded[df_exploded['Task'] == 'E2'].copy()
    E2_exploration_summary = (E2_df_exploded.groupby(['Subnum', 'Model', 'Condition', 'Task'])['exploration'].
                              value_counts().unstack(fill_value=0).reset_index())
    E2_exploration_summary['Exploration_Rate'] = E2_exploration_summary['exploration'] / 149
    E2_exploration_summary['Exploration_Rate_z'] = (E2_exploration_summary.groupby(['Model'])['Exploration_Rate'].
                                                    transform(lambda x: (x - x.mean()) / x.std()))

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
    # models = ['Delta', 'Decay', 'Decay_PVL']
    # for i, df in enumerate([delta_results, decay_results, decayPVL_results]):
    #     df['Condition'] = pd.Categorical(df['Condition'], categories=['Nature', 'Urban', 'Control'], ordered=True)
    #
    #
    # # Draw the model fitting results for all models and parameters
    # for i, model_result in enumerate([delta_results, decay_results, decayPVL_results]):
    #     for param in ['t', 'alpha', 'shape', 'la']:
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



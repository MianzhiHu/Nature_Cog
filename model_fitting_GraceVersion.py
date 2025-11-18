import os
import pandas as pd
from ComputationalModeling import (ComputationalModels, dict_generator, moving_window_model_fitting,
                                         parameter_extractor)

# ======================================================================================================================
# Load the data
# ======================================================================================================================
# print the number of cores available
print(f'Number of CPU cores available: {os.cpu_count()}')

dm_data = pd.read_csv('./dm_data.csv')
# For the purpose of 2025 Psychonomics, we only use IGT-SGT data
dm_data = dm_data[dm_data['Order'] == 'IGT_SGT'].copy()
dm_data['Trial'] = dm_data['Trial'].astype(int)
dm_data['keyResponse'] = dm_data['keyResponse'].astype(int)
dm_data['keyResponse'] = dm_data['keyResponse'] - 1
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

# delta, decay, delta_PVL, decay_PVL, delta_asymmetric, decay_PVPE
model_list = [delta_PVL, decay_PVL, decay_PVPE]
model_name_list = [name for name, obj in globals().items() if any(obj is m for m in model_list)]

# Set window parameters
window_size = 10
n_iterations = 100

if __name__== '__main__':
    # Fit SGT data overall
    for i, model in enumerate(model_list):
        SGT_result = model.fit(SGT_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha', num_exp_restart=100)
        SGT_result.to_csv(f'./SGT_{model_name_list[i]}.csv', index=False)
        IGT_result = model.fit(IGT_dict, num_iterations=n_iterations, initial_mode='first_trial_no_alpha', num_exp_restart=100)
        IGT_result.to_csv(f'./IGT_{model_name_list[i]}.csv', index=False)

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
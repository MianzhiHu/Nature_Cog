import os
import shutil
import numpy as np
import pandas as pd

# ======================================================================================================================
# Generate nature versus non-nature stimuli
# ======================================================================================================================
# define the paths
stimuli_path = './stimuli/all'
nature_stimuli_path = './stimuli/nature' # 226, 189
non_nature_stimuli_path = './stimuli/non_nature'
edge_stimuli_path = './stimuli/edge'

# create the folders if they do not exist
for path in [nature_stimuli_path, non_nature_stimuli_path, edge_stimuli_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# load stimuli info
stimuli_info = pd.read_csv('./stimuli/stimuli_info.csv')

# calculate the mean and standard deviation of the naturalness ratings
mean_naturalness = stimuli_info['Perc_Nat'].mean()
std_naturalness = stimuli_info['Perc_Nat'].std()
print(f'Mean naturalness: {mean_naturalness}, Standard deviation naturalness: {std_naturalness}')

# separate the stimuli into nature and non-nature
nat_threshold = stimuli_info['Perc_Nat'].quantile(0.75)
non_nat_threshold = stimuli_info['Perc_Nat'].quantile(0.25)

nature_stimuli_names = stimuli_info[stimuli_info['Perc_Nat'] >= nat_threshold]['ImageName'].values
non_nature_stimuli = stimuli_info[stimuli_info['Perc_Nat'] <= non_nat_threshold]['ImageName'].values

# Save the nature stimuli by name
for file in os.listdir(stimuli_path):
    file_name = file.split('.')[0]
    if file_name in nature_stimuli_names:
        src = os.path.join(stimuli_path, file)
        dst = os.path.join(nature_stimuli_path, file)
        shutil.copy(src, dst)
        print(f'{file_name} has been moved to nature folder at {dst}')
    elif file_name in non_nature_stimuli:
        src = os.path.join(stimuli_path, file)
        dst = os.path.join(non_nature_stimuli_path, file)
        shutil.copy(src, dst)
        print(f'{file_name} has been moved to non-nature folder at {dst}')
    elif 'edge' in file_name:
        # randomly select edge stimuli
        total_edge = len(os.listdir(edge_stimuli_path))
        if total_edge < len(nature_stimuli_names) and np.random.rand() < 0.3:
            src = os.path.join(stimuli_path, file)
            dst = os.path.join(edge_stimuli_path, file)
            shutil.copy(src, dst)
            print(f'{file_name} has been moved to edge folder at {dst}')
    else:
        print(f'{file_name} does not belong to either nature or non-nature category and was not selected for edge')

# print the total number of stimuli in each category
print(f'Total number of nature stimuli: {len(os.listdir(nature_stimuli_path))}')
print(f'Total number of non-nature stimuli: {len(os.listdir(non_nature_stimuli_path))}')
print(f'Total number of edge stimuli: {len(os.listdir(edge_stimuli_path))}')

# ======================================================================================================================
# Check the file names
# ======================================================================================================================
# Check the file names
nature_names = []
non_nature_names = []
edge_names = []
for file in os.listdir(nature_stimuli_path):
    nature_names.append(file)
for file in os.listdir(non_nature_stimuli_path):
    non_nature_names.append(file)
for file in os.listdir(edge_stimuli_path):
    edge_names.append(file)




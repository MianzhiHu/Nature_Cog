import numpy as np
import pandas as pd
import os
from utils.Between_Subj_Preprocessing_Zip import process_participant_data, determine_condition

# ======================================================================================================================
# Load the data
# ======================================================================================================================
all_participants_dfs = []
i = 0
main_folder_directory = ['./data/Data_25Spring/', './data/Data_25Summer/']
behavioral_list = ['React', 'Reward', 'keyResponse', 'Trial', 'Bank']
stimuli_info = pd.read_csv('./stimuli/stimuli_info.csv')

# Iterate over each subfolder in the main folder
for directory in main_folder_directory:
    for participant_folder_name in os.listdir(directory):
        print(f'Processing participant: {i + 1}')
        i += 1

        participant_folder_path = os.path.join(directory, participant_folder_name)

        # Check if this path is indeed a folder
        if os.path.isdir(participant_folder_path):
            # Process the participant folder and collect the DataFrame
            participant_df = process_participant_data(participant_folder_path, 1, 3, 2)
            participant_df['Subnum'] = i
            all_participants_dfs.append(participant_df)

# Drop the dfs that are empty
all_participants_dfs = [df for df in all_participants_dfs if not df.empty]

# Combine all participant DataFrames into one
all_data = pd.concat(all_participants_dfs, ignore_index=True)

# Insert a block number column with offset for later tasks
block_offset = np.where(all_data['Task'] == 'IGT', 10, 0)  # Add 10 to IGT blocks
all_data['Block'] = np.ceil(all_data['Trial'] / 10) + block_offset

# Detect the image rating task condition
img_conditions = all_data[all_data['Task'] == 'ImageRating'].groupby('Subnum').apply(determine_condition,
                                                                                     include_groups=False)
all_data['Condition'] = all_data['Subnum'].map(img_conditions)

# Detect if the participant selected the best option or the high frequency option
all_data['BestOption'] = all_data['keyResponse'].isin([3, 4]).astype(int)
all_data['HighFreqOption'] = ((all_data['Task'] == 'IGT') & all_data['keyResponse'].isin([2, 4]) |
                              (all_data['Task'] == 'SGT') & all_data['keyResponse'].isin([1, 2])).astype(int)

# Move the subject number and task columns to the front
for col_name in ['Condition', 'Task', 'Subnum']:
    col = all_data.pop(col_name)
    all_data.insert(0, col_name, col)

print(f'Currently, the total number of participants is {all_data["Subnum"].nunique()}')
print(f'Conditions: {all_data["Condition"].value_counts() // 250}')

# Save the data
img_data = all_data[all_data['Task'] == 'ImageRating'].dropna(axis=1, how='all')
dm_data = all_data[all_data['Task'] != 'ImageRating'].dropna(axis=1, how='all')

# Process stimuli information
stimuli_info.rename(columns={'ImageName': 'image_name'}, inplace=True)
img_data = img_data.merge(stimuli_info, on='image_name', how='left')
avg_rating = img_data.groupby(['Subnum']).agg({
    'Condition': 'first',
    'naturalness': 'mean',
    'disorderliness': 'mean',
    'aesthetic': 'mean',
    'Hue': 'mean',
    'Bright': 'mean',
    'Saturaton': 'mean',
    'SDhue': 'mean',
    'SDsat': 'mean',
    'Sdbright': 'mean',
    'Entropy': 'mean',
    'Perc_Nat': 'mean',
    'pc1f': 'mean',
    'pc2f': 'mean',
    'pc3f': 'mean',
    'pc4f': 'mean',
    'SED': 'mean',
    'total_ED': 'mean',
    'NSED': 'mean'
}).reset_index()
avg_rating.to_csv('./data/avg_rating.csv')

# Add stimuli information to the dm data
dm_data = dm_data.merge(avg_rating, on=['Subnum', 'Condition'], how='left')

# Save the data
all_data.to_csv('./data/all_data.csv', index=False)
img_data.to_csv('./data/img_data.csv', index=False)
dm_data.to_csv('./data/dm_data.csv', index=False)

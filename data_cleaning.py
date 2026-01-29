import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from utils.Between_Subj_Preprocessing_Zip import process_participant_data, determine_condition


def parse_d_hms(t):
    """
    Parse strings like '2:01:09:11' meaning D:HH:MM:SS.
    If format is shorter (e.g. HH:MM:SS), it still works.
    """
    parts = t.split(":")
    parts = list(map(int, parts))

    if len(parts) == 4:
        days, hours, minutes, seconds = parts
    elif len(parts) == 3:
        # HH:MM:SS
        days = 0
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unrecognized duration format: {t}")

    return pd.Timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

# ======================================================================================================================
# Load E1 Data
# ======================================================================================================================
all_participants_dfs = []
i = 0
SGT_IGT_folder_directory = ['./data/Data_SGT_IGT']
IGT_SGT_folder_directory = ['./data/IGT_SGT_Reduced']
behavioral_list = ['React', 'Reward', 'keyResponse', 'Trial', 'Bank']
# stimuli_info = pd.read_csv('./stimuli/stimuli_info.csv')
stimuli_info = pd.read_csv('./stimuli/visual_features_with_naturalness.csv')

# read json metadata
with open('./data/jatos_results_metadata_20260128160104.json', 'r') as f:
    metadata = json.load(f)

metadata = metadata['data'][0]['studyResults']

# for each participant, get their result ID and duration
result_ids = []
total_duration = []
duration_img = []
duration_2nd_task = []
for participant in metadata:
    result_ids.append(participant['id'])
    total_duration.append(participant['duration'])
    duration_img.append(participant['componentResults'][3]['duration'])
    duration_2nd_task.append(participant['componentResults'][4]['duration'])

metadata_duration = pd.DataFrame({
    'Result ID': result_ids,
    'Duration': total_duration,
    'Duration_ImageRating': duration_img,
    'Duration_2ndTask': duration_2nd_task
})

metadata_duration['Duration'] = metadata_duration['Duration'].apply(parse_d_hms)
metadata_duration['Duration_ImageRating'] = metadata_duration['Duration_ImageRating'].apply(parse_d_hms)
metadata_duration['Duration_2ndTask'] = metadata_duration['Duration_2ndTask'].apply(parse_d_hms)
metadata_duration['Duration_combined'] = metadata_duration['Duration_ImageRating'] + metadata_duration['Duration_2ndTask']

# First, remove all durations that are more than 1 hour
metadata_duration = metadata_duration[metadata_duration['Duration_combined'] < pd.Timedelta(hours=1)]
duration_mean = metadata_duration['Duration_combined'].mean()
duration_std = metadata_duration['Duration_combined'].std()
print(f'Maximum combined duration of Image Rating and 2nd Task is {metadata_duration["Duration_combined"].max()}.')
print(f'Average combined duration of Image Rating and 2nd Task is {duration_mean} with a std of {duration_std}.')
# Next, remove all durations that are more or less than 3 standard deviations from the mean
metadata_duration = metadata_duration[(metadata_duration['Duration_combined'] > duration_mean - 3 * duration_std) &
                                    (metadata_duration['Duration_combined'] < duration_mean + 3 * duration_std)]

# Plot the distribution of durations
plt.figure()
sns.histplot(metadata_duration['Duration_combined'].dt.total_seconds() / 60, bins=30, kde=True)
plt.title('Distribution of Task Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Participants')
sns.despine()
plt.savefig('./figures/Task_Duration_Distribution.png', dpi=600)
plt.close()

# Iterate over each subfolder in the main folder
for directory in SGT_IGT_folder_directory:
    for participant_folder_name in os.listdir(directory):
        # if the participant is not in the metadata, skip
        result_id = int(participant_folder_name.split('_')[2])
        if result_id not in metadata_duration['Result ID'].values:
            print(f'Skipping participant: {participant_folder_name} as they are not in the metadata.')
            continue

        print(f'Processing participant: {i + 1}')
        i += 1

        participant_folder_path = os.path.join(directory, participant_folder_name)

        # Check if this path is indeed a folder
        if os.path.isdir(participant_folder_path):
            num_folders = sum(os.path.isdir(os.path.join(participant_folder_path, name)) for name in os.listdir(participant_folder_path))
            if num_folders != 6:
                print(f'Participant {participant_folder_name} has {num_folders} folders, expected 6. Skipping.')
                continue
            # Process the participant folder and collect the DataFrame
            participant_df = process_participant_data(participant_folder_path, 1, 3, 2)
            participant_df['Subnum'] = i
            participant_df['Order'] = 'SGT_IGT'
            all_participants_dfs.append(participant_df)

# Now process the reversed order
for directory in IGT_SGT_folder_directory:
    for participant_folder_name in os.listdir(directory):
        # if the participant is not in the metadata, skip
        result_id = int(participant_folder_name.split('_')[2])
        if result_id not in metadata_duration['Result ID'].values:
            print(f'Skipping participant: {participant_folder_name} as they are not in the metadata.')
            continue

        print(f'Processing participant: {i + 1}')
        i += 1

        participant_folder_path = os.path.join(directory, participant_folder_name)

        # Check if this path is indeed a folder
        if os.path.isdir(participant_folder_path):
            num_folders = sum(os.path.isdir(os.path.join(participant_folder_path, name)) for name in os.listdir(participant_folder_path))
            if num_folders != 6:
                print(f'Participant {participant_folder_name} has {num_folders} folders, expected 6. Skipping.')
                continue
            # Process the participant folder and collect the DataFrame
            participant_df = process_participant_data(participant_folder_path, 3, 1, 2)
            participant_df['Subnum'] = i
            participant_df['Order'] = 'IGT_SGT'
            all_participants_dfs.append(participant_df)

# Drop the dfs that are empty
all_participants_dfs = [df for df in all_participants_dfs if not df.empty]

# Combine all participant DataFrames into one
all_data = pd.concat(all_participants_dfs, ignore_index=True)

# Insert a block number column with offset for later tasks
block_offset = np.where((all_data['Task'] == 'IGT') & (all_data['Order'] == 'SGT_IGT'), 10, 0)
block_offset += np.where((all_data['Task'] == 'SGT') & (all_data['Order'] == 'IGT_SGT'), 10, 0)
all_data['Block'] = np.ceil(all_data['Trial'] / 10) + block_offset

# Detect the image rating task condition
img_conditions = all_data[all_data['Task'] == 'ImageRating'].groupby('Subnum').apply(determine_condition,
                                                                                     include_groups=False)
all_data['Condition'] = all_data['Subnum'].map(img_conditions)

# Detect if the participant selected the best option or the high frequency option
all_data['BestOption'] = all_data['keyResponse'].isin([3, 4]).astype(int)
all_data['HighFreqOption'] = ((all_data['Task'] == 'IGT') & all_data['keyResponse'].isin([2, 4]) |
                              (all_data['Task'] == 'SGT') & all_data['keyResponse'].isin([1, 2])).astype(int)
all_data['HighMagOption'] = ((all_data['Task'] == 'IGT') & all_data['keyResponse'].isin([1, 2]) |
                             (all_data['Task'] == 'SGT') & all_data['keyResponse'].isin([1, 3])).astype(int)

# Move the subject number and task columns to the front
for col_name in ['Condition', 'Task', 'Subnum']:
    col = all_data.pop(col_name)
    all_data.insert(0, col_name, col)

#
conditions = [
    (all_data["Order"] == "IGT_SGT") & (all_data["Task"] == "IGT"),
    (all_data["Order"] == "IGT_SGT") & (all_data["Task"] == "SGT"),
    (all_data["Order"] == "SGT_IGT") & (all_data["Task"] == "IGT"),
    (all_data["Order"] == "SGT_IGT") & (all_data["Task"] == "SGT"),
]
task_labels = [1, 2, 2, 1]
all_data['TaskCode'] = np.select(conditions, task_labels, default=np.nan)

print(f'Currently, the total number of participants is {all_data["Subnum"].nunique()}')
print(f'[SGT-IGT] Conditions: {all_data[all_data["Order"] == "SGT_IGT"]["Condition"].value_counts() // 250} in SGT_IGT order')
print(f'[IGT-SGT] Conditions: {all_data[all_data["Order"] == "IGT_SGT"]["Condition"].value_counts() // 250} in IGT_SGT order')

# Save the data
img_data = all_data[all_data['Task'] == 'ImageRating'].dropna(axis=1, how='all')
dm_data = all_data[all_data['Task'] != 'ImageRating'].dropna(axis=1, how='all')

# Process stimuli information
stimuli_info.rename(columns={'ImageName': 'image_name'}, inplace=True)
img_data = img_data.merge(stimuli_info, on='image_name', how='left')
#
# # Find participants who rated all images the same
# n_unique_nat = img_data.groupby('Subnum')['naturalness'].nunique()
# n_unique_dis = img_data.groupby('Subnum')['disorderliness'].nunique()
# n_unique_aes = img_data.groupby('Subnum')['aesthetic'].nunique()
# constant_raters = n_unique_nat[(n_unique_nat == 1) | (n_unique_dis == 1) | (n_unique_aes == 1)].index.tolist()
#
# img_data = img_data[~img_data['Subnum'].isin(constant_raters)]
# print(f'Removed {len(constant_raters)} participants who rated all images the samely')
cols = img_data.columns.tolist()
visual_features = ['naturalness', 'disorderliness', 'aesthetic', 'Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton',
                   'SDSat', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'MeanTexture',
                   'SDTexture', 'Entropy', 'EdgeCount', 'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength',
                   'ContourSDLength', 'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH',
                   'KPMeanSize', 'KPSDSize', 'KPMeanStrength', 'KPSDStrength', 'KPMeanAngle', 'KPSDAngle', 'KPCount',
                   'sky', 'grass', 'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench', 'pole', 'building',
                   'tree', 'earth', 'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair', 'signboard', 'stairs',
                   'pot', 'sculpture', 'sidewalk', 'railing', 'road', 'person', 'mountain', 'lake', 'floor', 'car',
                   'traffic light', 'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']

# visual_features = ['naturalness', 'disorderliness', 'aesthetic', 'Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton',
#                    'SDSat', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'MeanTexture',
#                    'SDTexture', 'Entropy', 'EdgeCount', 'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength',
#                    'ContourSDLength', 'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH',
#                    'KPMeanSize', 'KPSDSize', 'KPMeanStrength', 'KPSDStrength', 'KPMeanAngle', 'KPSDAngle', 'KPCount',
#                    'sky', 'grass', 'plant', 'water', 'fence', 'path', 'river', 'bench', 'pole', 'building', 'tree',
#                    'earth', 'rock', 'streetlight', 'wall', 'signboard', 'sidewalk', 'railing', 'road', 'person',
#                    'mountain', 'car']

avg_rating = img_data.groupby(['Subnum'])[visual_features].mean().reset_index()
avg_rating.to_csv('./data/avg_rating.csv')

# Add stimuli information to the dm data
dm_data = dm_data.merge(avg_rating, on=['Subnum'], how='left')

# Detect inattentive participants
deck_counts = dm_data.groupby(['Subnum', 'Task'])['keyResponse'].nunique().reset_index()
deck_counts = deck_counts[deck_counts['keyResponse'] < 4]

# # get all participants who should be removed by combining constant raters and deck counts
# deck_counts = deck_counts[['Subnum']].drop_duplicates()
# deck_counts = pd.concat([deck_counts, pd.DataFrame({'Subnum': constant_raters})], ignore_index=True).drop_duplicates()

all_data = all_data[~all_data['Subnum'].isin(deck_counts['Subnum'])]
dm_data = dm_data[~dm_data['Subnum'].isin(deck_counts['Subnum'])]
dm_1a = dm_data[dm_data['Order'] == 'SGT_IGT']
dm_1b = dm_data[dm_data['Order'] == 'IGT_SGT']
img_data = img_data[~img_data['Subnum'].isin(deck_counts['Subnum'])]
print(f'After removing {deck_counts["Subnum"].nunique()} inattentive participants, the total number of participants is {all_data["Subnum"].nunique()}')
print(f'[SGT-IGT] Conditions: {all_data[all_data["Order"] == "SGT_IGT"]["Condition"].value_counts() // 250} in SGT_IGT order')
print(f'[IGT-SGT] Conditions: {all_data[all_data["Order"] == "IGT_SGT"]["Condition"].value_counts() // 250} in IGT_SGT order')

# Save the data
all_data.to_csv('./data/all_data.csv', index=False)
img_data.to_csv('./data/img_data.csv', index=False)
dm_data.to_csv('./data/dm_data.csv', index=False)
dm_1a.to_csv('./data/dm_1a.csv', index=False)
dm_1b.to_csv('./data/dm_1b.csv', index=False)

# ======================================================================================================================
# Load E2 Data
# ======================================================================================================================
E2_all_participants_dfs = []
i = 0
E2_folder_directory = ['./data/E2_Extended_Q']
behavioral_list = ['Trial', 'React', 'Reward', 'KeyResponse', 'BestChoice', 'EV_A', 'EV_B', 'EV_C', 'EV_D']

# Iterate over each subfolder in the main folder
for directory in E2_folder_directory:
    for participant_folder_name in os.listdir(directory):
        # if the participant is not in the metadata, skip
        result_id = int(participant_folder_name.split('_')[2])
        print(f'Processing participant: {i + 1}')
        i += 1

        participant_folder_path = os.path.join(directory, participant_folder_name)

        # Check if this path is indeed a folder
        if os.path.isdir(participant_folder_path):
            num_folders = sum(os.path.isdir(os.path.join(participant_folder_path, name)) for name in os.listdir(participant_folder_path))
            if num_folders != 6:
                print(f'Participant {participant_folder_name} has {num_folders} folders, expected 6. Skipping.')
                continue
            # Process the participant folder and collect the DataFrame
            participant_df = process_participant_data(participant_folder_path, 1, 3, 2)
            # if "Task" is sgt, change it to 1; if igt, change it to 2 because it is the same task now
            participant_df['Task'] = participant_df['Task'].replace({'SGT': 1, 'IGT': 2})
            participant_df['Subnum'] = i
            E2_all_participants_dfs.append(participant_df)

# Drop the dfs that are empty
E2_all_participants_dfs = [df for df in E2_all_participants_dfs if not df.empty]

# Combine all participant DataFrames into one
E2_all_data = pd.concat(E2_all_participants_dfs, ignore_index=True)

# Insert a block number column with offset for later tasks
E2_all_data['Block'] = np.ceil(E2_all_data['Trial'] / 10)

# Detect the image rating task condition
img_conditions = E2_all_data[E2_all_data['Task'] == 'ImageRating'].groupby('Subnum').apply(determine_condition,
                                                                                     include_groups=False)
E2_all_data['Condition'] = E2_all_data['Subnum'].map(img_conditions)

# Move the subject number and task columns to the front
for col_name in ['Condition', 'Task', 'Subnum']:
    col = E2_all_data.pop(col_name)
    E2_all_data.insert(0, col_name, col)

print(f'Currently, the total number of participants in E2 is {E2_all_data["Subnum"].nunique()}')
print(f'Conditions: {E2_all_data["Condition"].value_counts() // 350} in SGT_IGT order')

# Save the data
E2_img_data = E2_all_data[E2_all_data['Task'] == 'ImageRating'].dropna(axis=1, how='all')
E2_dm_data = E2_all_data[E2_all_data['Task'] != 'ImageRating'].dropna(axis=1, how='all')
E2_img_data = E2_img_data.merge(stimuli_info, on='image_name', how='left')

# Find participants who rated all images the same
E2_n_unique_nat = E2_img_data.groupby('Subnum')['naturalness'].nunique()
E2_n_unique_dis = E2_img_data.groupby('Subnum')['disorderliness'].nunique()
E2_n_unique_aes = E2_img_data.groupby('Subnum')['aesthetic'].nunique()
E2_constant_raters = E2_n_unique_nat[(E2_n_unique_nat == 1) | (E2_n_unique_dis == 1) | (E2_n_unique_aes == 1)].index.tolist()

E2_img_data = E2_img_data[~E2_img_data['Subnum'].isin(E2_constant_raters)]
print(f'Removed {len(E2_constant_raters)} participants who rated all images the samely')

E2_avg_rating = E2_img_data.groupby(['Subnum'])[visual_features].mean().reset_index()
E2_avg_rating.to_csv('./data/E2_avg_rating.csv')

# Add stimuli information to the dm data
E2_dm_data = E2_dm_data.merge(E2_avg_rating, on=['Subnum'], how='left')

# Detect inattentive participants
E2_deck_counts = E2_dm_data.groupby(['Subnum', 'Task'])['KeyResponse'].nunique().reset_index()
E2_deck_counts = E2_deck_counts[E2_deck_counts['KeyResponse'] < 4]

# get all participants who should be removed by combining constant raters and deck counts
E2_deck_counts = E2_deck_counts[['Subnum']].drop_duplicates()
E2_deck_counts = pd.concat([E2_deck_counts, pd.DataFrame({'Subnum': E2_constant_raters})], ignore_index=True).drop_duplicates()

E2_all_data = E2_all_data[~E2_all_data['Subnum'].isin(E2_deck_counts['Subnum'])]
E2_dm_data = E2_dm_data[~E2_dm_data['Subnum'].isin(E2_deck_counts['Subnum'])]
E2_img_data = E2_img_data[~E2_img_data['Subnum'].isin(E2_deck_counts['Subnum'])]
print(f'After removing {E2_deck_counts["Subnum"].nunique()} inattentive participants, the total number of participants is {E2_all_data["Subnum"].nunique()}')
print(f'Conditions: {E2_all_data["Condition"].value_counts() // 350} in SGT_IGT order')

# Save the data
E2_all_data.to_csv('./data/E2_all_data.csv', index=False)
E2_img_data.to_csv('./data/E2_img_data.csv', index=False)
E2_dm_data.to_csv('./data/E2_dm_data.csv', index=False)

# Quickly plot best choice rates per condition per task
best_choice_rate = E2_dm_data.groupby(['Subnum', 'Condition', 'Task'])['BestChoice'].mean().reset_index()
best_choice_rate_by_cond = best_choice_rate.groupby(['Condition', 'Task'])['BestChoice'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=best_choice_rate, x='Task', y='BestChoice', hue='Condition', errorbar='se')
plt.title('E2 Best Choice Rate by Condition and Task (Long Questionnaire)')
plt.ylabel('Best Choice Rate')
plt.xlabel('Task')
plt.ylim(0, 0.75)
sns.despine()
plt.savefig('./figures/E2_Best_Choice_Rate_by_Condition_and_Task.png', dpi=600)
plt.close()
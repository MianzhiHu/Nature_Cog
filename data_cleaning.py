import ast
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
# Load Daw Extended Q data
# ======================================================================================================================
# Load stimuli info
stimuli_info = pd.read_csv('./stimuli/visual_features_with_naturalness.csv')
stimuli_info.rename(columns={'ImageName': 'image_name'}, inplace=True)
rating_features = ['naturalness', 'disorderliness', 'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery',
                   'imagability', 'control']
visual_features = ['naturalness', 'disorderliness', 'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery',
                   'imagability', 'control', 'Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton', 'SDSat', 'Contrast',
                   'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'MeanTexture', 'SDTexture', 'Entropy',
                   'EdgeCount', 'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength', 'ContourSDLength',
                   'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH', 'KPMeanSize',
                   'KPSDSize', 'KPMeanStrength', 'KPSDStrength', 'KPMeanAngle', 'KPSDAngle', 'KPCount', 'sky', 'grass',
                   'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench', 'pole', 'building', 'tree', 'earth',
                   'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair', 'signboard', 'stairs', 'pot', 'sculpture',
                   'sidewalk', 'railing', 'road', 'person', 'mountain', 'lake', 'floor', 'car', 'traffic light',
                   'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']


E1_all_participants_dfs = []
E1_folder_directory = ['./data/Daw_Extended_Q']
behavioral_list = ['Trial', 'React', 'Reward', 'KeyResponse', 'BestChoice', 'EV_A', 'EV_B', 'EV_C', 'EV_D']
i = 0

# Iterate over each subfolder in the main folder
for directory in E1_folder_directory:
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
            E1_all_participants_dfs.append(participant_df)

# Drop the dfs that are empty
E1_all_participants_dfs = [df for df in E1_all_participants_dfs if not df.empty]

# Combine all participant DataFrames into one
E1_all_data = pd.concat(E1_all_participants_dfs, ignore_index=True)

# Insert a block number column with offset for later tasks
E1_all_data['Block'] = np.ceil(E1_all_data['Trial'] / 10)

# If Age is not a number, put as NaN
E1_all_data['Age'] = pd.to_numeric(E1_all_data['Age'], errors='coerce')

# Detect the image rating task condition
img_conditions = E1_all_data[E1_all_data['Task'] == 'ImageRating'].groupby('Subnum').apply(determine_condition,
                                                                                     include_groups=False)
E1_all_data['Condition'] = E1_all_data['Subnum'].map(img_conditions)

# Move the subject number and task columns to the front
for col_name in ['Condition', 'Task', 'Subnum']:
    col = E1_all_data.pop(col_name)
    E1_all_data.insert(0, col_name, col)

print(f'Currently, the total number of participants in E1 is {E1_all_data["Subnum"].nunique()}')
print(f'Conditions: {E1_all_data["Condition"].value_counts() // 350}')

# Save the data
E1_img_data = E1_all_data[E1_all_data['Task'] == 'ImageRating'].dropna(axis=1, how='all')
E1_dm_data = E1_all_data[E1_all_data['Task'] != 'ImageRating'].dropna(axis=1, how='all')
E1_img_data = E1_img_data.merge(stimuli_info, on='image_name', how='left')

# Find participants who rated all images the same
rating_cols = ['naturalness', 'disorderliness', 'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery', 'imagability', 'control']
n_unique = E1_img_data.groupby('Subnum')[rating_cols].nunique()
E1_constant_raters = n_unique.index[(n_unique == 1).any(axis=1)].tolist()

E1_img_data = E1_img_data[~E1_img_data['Subnum'].isin(E1_constant_raters)]
print(f'Removed {len(E1_constant_raters)} participants who rated all images the samely')

E1_avg_rating = E1_img_data.groupby(['Subnum'])[visual_features].mean().reset_index()
E1_avg_rating.to_csv('./data/E1_avg_rating.csv')

# Add stimuli information to the dm data
E1_dm_data = E1_dm_data.merge(E1_avg_rating, on=['Subnum'], how='left')

# Detect inattentive participants
E1_deck_counts = E1_dm_data.groupby(['Subnum', 'Task'])['KeyResponse'].nunique().reset_index()
E1_deck_counts = E1_deck_counts[E1_deck_counts['KeyResponse'] < 4]

# get all participants who should be removed by combining constant raters and deck counts
E1_deck_counts = E1_deck_counts[['Subnum']].drop_duplicates()
E1_deck_counts = pd.concat([E1_deck_counts, pd.DataFrame({'Subnum': E1_constant_raters})], ignore_index=True).drop_duplicates()

E1_all_data = E1_all_data[~E1_all_data['Subnum'].isin(E1_deck_counts['Subnum'])]
E1_dm_data = E1_dm_data[~E1_dm_data['Subnum'].isin(E1_deck_counts['Subnum'])]
E1_img_data = E1_img_data[~E1_img_data['Subnum'].isin(E1_deck_counts['Subnum'])]
print(f'After removing {E1_deck_counts["Subnum"].nunique()} inattentive participants, the total number of participants is {E1_all_data["Subnum"].nunique()}')
print(f'Conditions: {E1_all_data["Condition"].value_counts() // 350}')

# Save the data
E1_all_data.to_csv('./data/E1_all_data.csv', index=False)
E1_img_data.to_csv('./data/E1_img_data.csv', index=False)
E1_dm_data.to_csv('./data/E1_dm_data.csv', index=False)

# Now E2
E2_all_participants_dfs = []
E2_folder_directory = ['./data/Daw_Background']
behavioral_list = ['Trial', 'React', 'Reward', 'KeyResponse', 'BestChoice', 'EV_A', 'EV_B', 'EV_C', 'EV_D']
i = 0

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
            if num_folders < 3:
                print(f'Participant {participant_folder_name} has {num_folders} folders, expected 6. Skipping.')
                continue
            dfs = []
            # iterate each sub‑folder
            for folder_name in os.listdir(participant_folder_path):
                folder_path = os.path.join(participant_folder_path, folder_name)

                # load each .txt as JSON lines
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for lineno, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                # skip empty lines
                                continue
                            try:
                                dfs.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

            # remove the empty dfs
            dfs = [df for df in dfs if df]

            # extract demo info
            demo_info = {k: ast.literal_eval(v) for k, v in dfs[0].items()}
            demo_info = pd.DataFrame(demo_info).reset_index(drop=True)

            # change all the dfs to DataFrame
            try:
                dfs[1] = pd.DataFrame(dfs[1])
            except Exception as e:
                print(f'Error processing participant {participant_folder_name}: {e}')
                continue

            # combine all the dfs into one
            df = pd.concat([demo_info, dfs[1]], ignore_index=True)
            df[['Gender', 'Ethnicity', 'Race', 'Age']] = df[['Gender', 'Ethnicity', 'Race', 'Age']].ffill()
            df = df.iloc[1:]

            # # Process the participant folder and collect the DataFrame
            # participant_df = process_participant_data(participant_folder_path, 0, 999, 1)
            # # if "Task" is sgt, change it to 1; if igt, change it to 2 because it is the same task now
            # participant_df['Task'] = participant_df['Task'].replace({'SGT': 1, 'IGT': 2})
            df['Subnum'] = i
            E2_all_participants_dfs.append(df)

# Drop the dfs that are empty
E2_all_participants_dfs = [df for df in E2_all_participants_dfs if not df.empty]

# Combine all participant DataFrames into one
E2_all_data = pd.concat(E2_all_participants_dfs, ignore_index=True)

# Insert a block number column with offset for later tasks
E2_all_data['Block'] = np.ceil(E2_all_data['Trial'] / 10)

# Detect missing age info
E2_all_data['Age'] = pd.to_numeric(E2_all_data['Age'], errors='coerce')

# Detect the condition
img_conditions = E2_all_data.groupby('Subnum').apply(lambda group: determine_condition(group, col_name='ImageNames'),
                                                     include_groups=False)
E2_all_data['Condition'] = E2_all_data['Subnum'].map(img_conditions)

# Move the subject number and task columns to the front
for col_name in ['Block', 'Condition', 'Subnum']:
    col = E2_all_data.pop(col_name)
    E2_all_data.insert(0, col_name, col)

print(f'Currently, the total number of participants in E2 is {E2_all_data["Subnum"].nunique()}')
print(f'Conditions: {E2_all_data["Condition"].value_counts() // 250}')

# Save the data
E2_all_data.rename(columns={'ImageNames': 'image_name'}, inplace=True)
E2_all_data = E2_all_data.merge(stimuli_info, on='image_name', how='left')

# Calculate the average rating for each picture from E1
E1_ratings = E1_img_data.groupby('image_name')[rating_features].mean().reset_index()
E2_all_data = E2_all_data.merge(E1_ratings, on='image_name', how='left')

# Remove ratings from visual features
E2_avg_rating = E2_all_data.groupby(['Subnum'])[visual_features].mean().reset_index()
E2_avg_rating.to_csv('./data/E2_avg_rating.csv')

# Add stimuli information to the dm data
E2_all_data = E2_all_data.merge(E2_avg_rating, on=['Subnum'], how='left')

# Detect inattentive participants
E2_deck_counts = E2_all_data.groupby(['Subnum'])['KeyResponse'].nunique().reset_index()
E2_deck_counts = E2_deck_counts[E2_deck_counts['KeyResponse'] < 4]
E2_deck_counts = E2_deck_counts[['Subnum']].drop_duplicates()

E2_all_data = E2_all_data[~E2_all_data['Subnum'].isin(E2_deck_counts['Subnum'])]
print(f'After removing {E2_deck_counts["Subnum"].nunique()} inattentive participants, the total number of participants is {E2_all_data["Subnum"].nunique()}')
print(f'Conditions: {E2_all_data["Condition"].value_counts() // 250}')

# Save the data
E2_all_data.to_csv('./data/E2_all_data.csv', index=False)
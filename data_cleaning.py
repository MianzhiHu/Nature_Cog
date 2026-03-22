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
visual_features = ['naturalness', 'disorderliness', 'aesthetic', 'Hue', 'SDHue', 'Bright', 'SDBright', 'Saturaton',
                   'SDSat', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'MeanTexture',
                   'SDTexture', 'Entropy', 'EdgeCount', 'CornerMean', 'CornerSD', 'CornerCount', 'ContourMeanLength',
                   'ContourSDLength', 'ContourMeanArea', 'ContourSDArea', 'ContourCount', 'AsymmetryV', 'AsymmetryH',
                   'KPMeanSize', 'KPSDSize', 'KPMeanStrength', 'KPSDStrength', 'KPMeanAngle', 'KPSDAngle', 'KPCount',
                   'sky', 'grass', 'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench', 'pole', 'building',
                   'tree', 'earth', 'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair', 'signboard', 'stairs',
                   'pot', 'sculpture', 'sidewalk', 'railing', 'road', 'person', 'mountain', 'lake', 'floor', 'car',
                   'traffic light', 'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']


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

# Detect the image rating task condition
img_conditions = E1_all_data[E1_all_data['Task'] == 'ImageRating'].groupby('Subnum').apply(determine_condition,
                                                                                     include_groups=False)
E1_all_data['Condition'] = E1_all_data['Subnum'].map(img_conditions)

# Move the subject number and task columns to the front
for col_name in ['Condition', 'Task', 'Subnum']:
    col = E1_all_data.pop(col_name)
    E1_all_data.insert(0, col_name, col)

print(f'Currently, the total number of participants in E1 is {E1_all_data["Subnum"].nunique()}')
print(f'Conditions: {E1_all_data["Condition"].value_counts() // 350} in SGT_IGT order')

# Save the data
E1_img_data = E1_all_data[E1_all_data['Task'] == 'ImageRating'].dropna(axis=1, how='all')
E1_dm_data = E1_all_data[E1_all_data['Task'] != 'ImageRating'].dropna(axis=1, how='all')
E1_img_data = E1_img_data.merge(stimuli_info, on='image_name', how='left')

# Find participants who rated all images the same
E1_n_unique_nat = E1_img_data.groupby('Subnum')['naturalness'].nunique()
E1_n_unique_dis = E1_img_data.groupby('Subnum')['disorderliness'].nunique()
E1_n_unique_aes = E1_img_data.groupby('Subnum')['aesthetic'].nunique()
E1_constant_raters = E1_n_unique_nat[(E1_n_unique_nat == 1) | (E1_n_unique_dis == 1) | (E1_n_unique_aes == 1)].index.tolist()

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
print(f'Conditions: {E1_all_data["Condition"].value_counts() // 350} in SGT_IGT order')

# Save the data
E1_all_data.to_csv('./data/E1_all_data.csv', index=False)
E1_img_data.to_csv('./data/E1_img_data.csv', index=False)
E1_dm_data.to_csv('./data/E1_dm_data.csv', index=False)

# Quickly plot best choice rates per condition per task
print(E1_dm_data['React'].mean())
best_choice_rate = E1_dm_data.groupby(['Subnum', 'Condition', 'Task'])['BestChoice'].mean().reset_index()
# take z-score
best_choice_rate_z = best_choice_rate.copy()
best_choice_rate_z['BestChoice'] = best_choice_rate.groupby(['Task'])['BestChoice'].transform(lambda x: (x - x.mean()) / x.std())
best_choice_rate_diff = best_choice_rate_z.pivot(index=['Subnum', 'Condition'], columns='Task', values='BestChoice').reset_index()
best_choice_rate_diff['BestChoice_Diff'] = best_choice_rate_diff[2] - best_choice_rate_diff[1]
best_choice_rate_by_cond = best_choice_rate.groupby(['Condition', 'Task'])['BestChoice'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=best_choice_rate_z, x='Task', y='BestChoice', hue='Condition', errorbar='se')
plt.title('E1 Best Choice Rate by Condition and Task (Long Questionnaire)')
plt.ylabel('Best Choice Rate')
plt.xlabel('Task')
sns.despine()
plt.savefig('./figures/E1_Best_Choice_Rate_by_Condition_and_Task.png', dpi=600)
plt.close()

plt.figure(figsize=(8, 6))
sns.barplot(data=best_choice_rate_diff, x='Condition', y='BestChoice_Diff', errorbar='se')
plt.title('E1 Best Choice Rate by Condition and Task (Long Questionnaire)')
plt.ylabel('Best Choice Rate')
plt.xlabel('Task')
sns.despine()
plt.savefig('./figures/E1_Best_Choice_Diff_Rate_by_Condition_and_Task.png', dpi=600)
plt.close()


# anova
import pingouin as pg
anova = pg.anova(dv='BestChoice_Diff', between='Condition', data=best_choice_rate_diff, detailed=True)
pairwise = pg.pairwise_tests(dv='BestChoice_Diff', between='Condition', data=best_choice_rate_diff, padjust='fdr_bh')
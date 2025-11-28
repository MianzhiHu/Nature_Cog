import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Load image data
img_data = pd.read_csv('./stimuli/visual_features_extracted.csv')
print(img_data.columns)
semantic_cols = ['Category', 'ImageName', 'sky', 'grass', 'plant', 'water', 'sea', 'fence', 'path', 'river', 'bench',
                 'pole', 'building', 'tree', 'earth', 'rock', 'streetlight', 'ashcan', 'table', 'wall', 'chair',
                 'signboard', 'stairs', 'pot', 'sculpture', 'sidewalk', 'railing', 'road', 'person', 'mountain',
                 'lake', 'floor', 'car', 'traffic light']
semantic_data = img_data[semantic_cols].copy()

# Perform PCA on semantic features
scaler = StandardScaler()
semantic_data_scaled = scaler.fit_transform(semantic_data.drop(columns=['Category', 'ImageName']))
pca = PCA(n_components=1)
semantic_pca = pca.fit_transform(semantic_data_scaled)
semantic_pca_df = pd.DataFrame(semantic_pca, columns=[f'Semantic_PC{i+1}' for i in range(semantic_pca.shape[1])])
semantic_pca_df['Category'] = semantic_data['Category'].values
semantic_pca_df['ImageName'] = semantic_data['ImageName'].values

# process PCA results
semantic_pca_df['Semantic_PC1'] = - semantic_pca_df['Semantic_PC1'] # flip the sign for interpretability
semantic_pca_df['Naturalness_PCA'] = ((semantic_pca_df['Semantic_PC1'] - semantic_pca_df['Semantic_PC1'].min()) /
                                      (semantic_pca_df['Semantic_PC1'].max() - semantic_pca_df['Semantic_PC1'].min()))

# save the PCA results
img_data = pd.merge(img_data, semantic_pca_df[['ImageName', 'Naturalness_PCA', 'Semantic_PC1']], on='ImageName', how='left')
img_data.to_csv('./stimuli/visual_features_with_naturalness.csv', index=False)

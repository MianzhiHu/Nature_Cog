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

# run tests to see how many components to keep
scaler = StandardScaler()
semantic_data_scaled = scaler.fit_transform(semantic_data.drop(columns=['Category', 'ImageName']))
pca_test = PCA().fit(semantic_data_scaled)
explained_variance = pca_test.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Test eigenvalues greater than 1 rule
eigenvalues = pca_test.explained_variance_
num_components_eigen_gt_1 = np.sum(eigenvalues > 1)
print(f'Number of components with eigenvalues > 1: {num_components_eigen_gt_1}')

# parallel analysis
n_samples, n_features = semantic_data_scaled.shape
n_iterations = 10000
random_eigenvalues = np.zeros((n_iterations, n_features))
for i in range(n_iterations):
    random_data = np.random.normal(size=(n_samples, n_features))
    pca_random = PCA().fit(random_data)
    random_eigenvalues[i, :] = pca_random.explained_variance_
percentile_95 = np.percentile(random_eigenvalues, 95, axis=0)
overall_percentile_95 = np.percentile(random_eigenvalues, 95)

# Scree plot with parallel analysis
plt.figure()
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', label='Actual Data')
plt.plot(range(1, len(percentile_95) + 1), percentile_95, marker='o', label='95th Percentile Random Data')
plt.title('Parallel Analysis')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1 Threshold')
plt.axhline(y=overall_percentile_95, color='g', linestyle='--', label='Overall 95th Percentile Threshold')
plt.legend()
plt.grid()
plt.savefig('./figures/semantic_features_parallel_analysis.png', dpi=600)
plt.show()

# N = 3

# Perform PCA on semantic features
pca = PCA(n_components=3)
semantic_pca = pca.fit_transform(semantic_data_scaled)
semantic_pca_df = pd.DataFrame(semantic_pca, columns=[f'Semantic_PC{i+1}' for i in range(semantic_pca.shape[1])])
semantic_pca_df['Category'] = semantic_data['Category'].values
semantic_pca_df['ImageName'] = semantic_data['ImageName'].values

# Extract PCA loadings
loadings = pca.components_.T
loading_df = pd.DataFrame(loadings, index=semantic_data.columns[2:], columns=[f'Semantic_PC{i+1}' for i in range(semantic_pca.shape[1])])
loading_df.to_csv('./stimuli/semantic_feature_pca_loadings.csv')

# # process PCA results
# semantic_pca_df['Semantic_PC1'] = - semantic_pca_df['Semantic_PC1'] # flip the sign for interpretability
# semantic_pca_df['Naturalness_PCA'] = ((semantic_pca_df['Semantic_PC1'] - semantic_pca_df['Semantic_PC1'].min()) /
#                                       (semantic_pca_df['Semantic_PC1'].max() - semantic_pca_df['Semantic_PC1'].min()))

# save the PCA results
img_data = pd.merge(img_data, semantic_pca_df[['ImageName', 'Semantic_PC1', 'Semantic_PC2', 'Semantic_PC3']], on='ImageName', how='left')
img_data.to_csv('./stimuli/visual_features_with_naturalness.csv', index=False)

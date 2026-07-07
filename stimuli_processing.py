import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from skimage.feature import graycomatrix, graycoprops
from scipy import stats
from transformers import AutoTokenizer, AutoModelForSemanticSegmentation, AutoProcessor
from PIL import Image
import torch

# # ======================================================================================================================
# # Generate nature versus non-nature stimuli
# # ======================================================================================================================
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

# ======================================================================================================================
# Correlate E1 image-wise naturalness ratings with original Perc_Nature
# ======================================================================================================================
perc_nature_col = 'Perc_Nature' if 'Perc_Nature' in stimuli_info.columns else 'Perc_Nat'

E1_image_ratings = pd.read_csv('./data/E1_img_data.csv', usecols=['image_name', 'naturalness'])
E1_image_ratings['ImageName'] = E1_image_ratings['image_name'].astype(str).str.replace(r'\.[^.]+$', '', regex=True)
E1_image_ratings['naturalness'] = pd.to_numeric(E1_image_ratings['naturalness'], errors='coerce')

E1_image_naturalness = (
    E1_image_ratings
    .dropna(subset=['ImageName', 'naturalness'])
    .groupby('ImageName', as_index=False)
    .agg(
        E1_naturalness=('naturalness', 'mean'),
        n_E1_ratings=('naturalness', 'count')
    )
)

E1_naturalness_perc_nature = E1_image_naturalness.merge(
    stimuli_info[['ImageName', perc_nature_col]],
    on='ImageName',
    how='inner'
)
E1_naturalness_perc_nature[perc_nature_col] = pd.to_numeric(
    E1_naturalness_perc_nature[perc_nature_col],
    errors='coerce'
)
E1_naturalness_perc_nature = E1_naturalness_perc_nature.dropna(
    subset=['E1_naturalness', perc_nature_col]
)
low_perc_nature_threshold = stimuli_info[perc_nature_col].quantile(0.25)
high_perc_nature_threshold = stimuli_info[perc_nature_col].quantile(0.75)
E1_naturalness_perc_nature['Original_Group'] = np.select(
    [
        E1_naturalness_perc_nature[perc_nature_col] <= low_perc_nature_threshold,
        E1_naturalness_perc_nature[perc_nature_col] >= high_perc_nature_threshold
    ],
    ['Urban-selected', 'Nature-selected'],
    default='Middle'
)

if len(E1_naturalness_perc_nature) >= 2:
    r, p = stats.pearsonr(
        E1_naturalness_perc_nature['E1_naturalness'],
        E1_naturalness_perc_nature[perc_nature_col]
    )
else:
    r, p = np.nan, np.nan

E1_naturalness_perc_nature_corr = pd.DataFrame({
    'rating_variable': ['E1_naturalness'],
    'stimuli_variable': [perc_nature_col],
    'n_images': [len(E1_naturalness_perc_nature)],
    'pearson_r': [r],
    'p_value': [p]
})

E1_naturalness_perc_nature.to_csv('./stimuli/E1_image_naturalness_vs_perc_nature.csv', index=False)
E1_naturalness_perc_nature_corr.to_csv('./stimuli/E1_naturalness_perc_nature_correlation.csv', index=False)

print(
    f"E1 image-wise naturalness vs {perc_nature_col}: "
    f"r = {r:.3f}"
)

font_path = 'utils/AbhayaLibre-ExtraBold.ttf'
plot_font = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
x = E1_naturalness_perc_nature['E1_naturalness']
y = E1_naturalness_perc_nature[perc_nature_col]

group_colors = {
    'Urban-selected': '#C44E52',
    'Nature-selected': '#55A868',
    'Middle': '#7F7F7F'
}

for group_name, group_data in E1_naturalness_perc_nature.groupby('Original_Group'):
    ax.scatter(
        group_data['E1_naturalness'],
        group_data[perc_nature_col],
        s=60,
        color=group_colors[group_name],
        alpha=0.78,
        edgecolor='white',
        linewidth=0.7,
        label=group_name,
        zorder=2
    )

if len(E1_naturalness_perc_nature) >= 2:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color='#222222',
        linewidth=2.5,
        zorder=3
    )

ax.text(
    0.97, 0.05,
    rf"$r$ = {r:.2f}",
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=12,
    fontproperties=plot_font,
    bbox=dict(boxstyle='round,pad=0.30', facecolor='white', edgecolor='0.85', alpha=0.92)
)

ax.set_xlabel('E1 Naturalness Rating', fontsize=14, fontproperties=plot_font)
ax.set_ylabel(f'Original Naturalness ({perc_nature_col})', fontsize=14, fontproperties=plot_font)

for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
    tick_label.set_fontproperties(plot_font)
    tick_label.set_fontsize(11)

legend = ax.legend(
    title='Original Quartile',
    loc='upper left',
    frameon=True,
    facecolor='white',
    edgecolor='0.85'
)
legend.get_title().set_fontproperties(plot_font)
legend.get_title().set_fontsize(11)
for text in legend.get_texts():
    text.set_fontproperties(plot_font)
    text.set_fontsize(10)

ax.tick_params(axis='both', width=1.2, length=5)
ax.grid(axis='both', color='0.90', linewidth=0.8, zorder=0)

plt.tight_layout()
os.makedirs('./figures', exist_ok=True)
fig.savefig('./figures/E1_naturalness_vs_perc_nature.png', dpi=600, bbox_inches='tight')
plt.close(fig)

# # calculate the mean and standard deviation of the naturalness ratings
# mean_naturalness = stimuli_info['Perc_Nat'].mean()
# std_naturalness = stimuli_info['Perc_Nat'].std()
# print(f'Mean naturalness: {mean_naturalness}, Standard deviation naturalness: {std_naturalness}')
#
# # separate the stimuli into nature and non-nature
# nat_threshold = stimuli_info['Perc_Nat'].quantile(0.75)
# non_nat_threshold = stimuli_info['Perc_Nat'].quantile(0.25)
#
# nature_stimuli_names = stimuli_info[stimuli_info['Perc_Nat'] >= nat_threshold]['ImageName'].values
# non_nature_stimuli = stimuli_info[stimuli_info['Perc_Nat'] <= non_nat_threshold]['ImageName'].values
#
# # Save the nature stimuli by name
# for file in os.listdir(stimuli_path):
#     file_name = file.split('.')[0]
#     if file_name in nature_stimuli_names:
#         src = os.path.join(stimuli_path, file)
#         dst = os.path.join(nature_stimuli_path, file)
#         shutil.copy(src, dst)
#         print(f'{file_name} has been moved to nature folder at {dst}')
#     elif file_name in non_nature_stimuli:
#         src = os.path.join(stimuli_path, file)
#         dst = os.path.join(non_nature_stimuli_path, file)
#         shutil.copy(src, dst)
#         print(f'{file_name} has been moved to non-nature folder at {dst}')
#     elif 'edge' in file_name:
#         # randomly select edge stimuli
#         total_edge = len(os.listdir(edge_stimuli_path))
#         if total_edge < len(nature_stimuli_names) and np.random.rand() < 0.3:
#             src = os.path.join(stimuli_path, file)
#             dst = os.path.join(edge_stimuli_path, file)
#             shutil.copy(src, dst)
#             print(f'{file_name} has been moved to edge folder at {dst}')
#     else:
#         print(f'{file_name} does not belong to either nature or non-nature category and was not selected for edge')
#
# # print the total number of stimuli in each category
# print(f'Total number of nature stimuli: {len(os.listdir(nature_stimuli_path))}')
# print(f'Total number of non-nature stimuli: {len(os.listdir(non_nature_stimuli_path))}')
# print(f'Total number of edge stimuli: {len(os.listdir(edge_stimuli_path))}')

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

# ======================================================================================================================
# Load all images
# ======================================================================================================================
distances = [1, 2, 4, 8]
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images[filename.split('.')[0]] = img
    return images

nature_images = load_images_from_folder(nature_stimuli_path)
non_nature_images = load_images_from_folder(non_nature_stimuli_path)
edge_images = load_images_from_folder(edge_stimuli_path)
print(f'Loaded {len(nature_images)} nature images, {len(non_nature_images)} non-nature images, and {len(edge_images)} edge images.')

# Calculate low-level visual features for each category
def extract_visual_features(pathm, upper_threshold=0.60, lower_threshold=0.25):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_pixels = h * w
    H = hsv[:, :, 0].astype(np.float64)
    S = hsv[:, :, 1].astype(np.float64)
    V = hsv[:, :, 2].astype(np.float64)

    # hue stats
    mean_hue = float(H.mean())
    SDhue = float(H.std())

    # brightness stats
    Bright = float(V.mean())
    Sdbright = float(V.std())

    # saturation stats
    Saturation = float(S.mean())
    SDsat = float(S.std())

    # texture features
    glcm = graycomatrix(gray, distances=distances, angles=angles)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    mean_texture = graycoprops(glcm, 'mean').flatten()
    std_texture = graycoprops(glcm, 'std').flatten()
    entropy = graycoprops(glcm, 'entropy').flatten()

    # edges
    upper = upper_threshold * gray.max()
    lower = lower_threshold * gray.max()
    edges = cv2.Canny(gray, lower, upper)

    ED = float(np.count_nonzero(edges)) / total_pixels

    # features
    sift = cv2.SIFT_create()
    keypoints, _  = sift.detectAndCompute(gray, None)
    kp_size = np.array([kp.size for kp in keypoints]).mean()
    kp_size_sd = np.array([kp.size for kp in keypoints]).std()
    kp_strength = np.array([kp.response for kp in keypoints]).mean()
    kp_strength_sd = np.array([kp.response for kp in keypoints]).std()
    kp_angles = np.array([kp.angle for kp in keypoints]).mean()
    kp_angles_sd = np.array([kp.angle for kp in keypoints]).std()
    kp_length = len(keypoints)

    # contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_lengths = [cv2.arcLength(cnt, True) for cnt in contours]
    countor_area = [cv2.contourArea(cnt) for cnt in contours]
    mean_contour_length = np.mean(contour_lengths)
    sd_contour_length = np.std(contour_lengths)
    mean_contour_area = np.mean(countor_area)
    sd_contour_area = np.std(countor_area)
    contour_length = len(contours)

    # corners
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=10000, qualityLevel=0.01, minDistance=10)
    corner_strengths = [gray[int(c[0][1]), int(c[0][0])] for c in corners]
    mean_corner_strength = np.mean(corner_strengths)
    sd_corner_strength = np.std(corner_strengths)
    corner_count = len(corners)

    # vertical symmetry
    flipped_img = cv2.flip(gray, 1)
    asymmetry_v = float(np.sum(cv2.absdiff(gray, flipped_img))) / total_pixels

    # horizontal symmetry
    flipped_img_h = cv2.flip(gray, 0)
    asymmetry_h = float(np.sum(cv2.absdiff(gray, flipped_img_h))) / total_pixels

    return {
        "Hue": mean_hue,
        "SDHue": SDhue,
        "Bright": Bright,
        "SDBright": Sdbright,
        "Saturaton": Saturation,
        "SDSat": SDsat,
        "Contrast": contrast.mean(),
        "Dissimilarity": dissimilarity.mean(),
        "Homogeneity": homogeneity.mean(),
        "Energy": energy.mean(),
        "Correlation": correlation.mean(),
        "MeanTexture": mean_texture.mean(),
        "SDTexture": std_texture.mean(),
        "Entropy": entropy.mean(),
        "EdgeCount": ED,
        "CornerMean": mean_corner_strength,
        "CornerSD": sd_corner_strength,
        "CornerCount": corner_count,
        "ContourMeanLength": mean_contour_length,
        "ContourSDLength": sd_contour_length,
        "ContourMeanArea": mean_contour_area,
        "ContourSDArea": sd_contour_area,
        "ContourCount": contour_length,
        "AsymmetryV": asymmetry_v,
        "AsymmetryH": asymmetry_h,
        "KPMeanSize": kp_size,
        "KPSDSize": kp_size_sd,
        "KPMeanStrength": kp_strength,
        "KPSDStrength": kp_strength_sd,
        "KPMeanAngle": kp_angles,
        "KPSDAngle": kp_angles_sd,
        "KPCount": kp_length
    }

features_reconstructed = []
for dir in [nature_stimuli_path, non_nature_stimuli_path, edge_stimuli_path]:
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        features = extract_visual_features(path)
        features['ImageName'] = file.split('.')[0]
        features_reconstructed.append(features)

visual_features_df = pd.DataFrame(features_reconstructed)

# ======================================================================================================================
# Semantic segmentation using SegFormer
# ======================================================================================================================
model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
feature_extractor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
model.eval()
id2label = model.config.id2label  # dict: {0: 'wall', 1: 'building', ...}
features_extracted = []

for path in [nature_stimuli_path, non_nature_stimuli_path]:
    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Get segmentation map
        seg_map = logits.argmax(dim=1)[0].cpu().numpy()
        unique_ids, counts = np.unique(seg_map, return_counts=True)

        area_fractions = {
            id2label[class_id]: counts[idx] / seg_map.size
            for idx, class_id in enumerate(unique_ids)
        }

        area_fractions['ImageName'] = file.split('.')[0]
        area_fractions['Category'] = 'Nature' if path == nature_stimuli_path else 'Urban'
        features_extracted.append(area_fractions)

features_df = pd.DataFrame(features_extracted)
cols = features_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('ImageName'))) # make the ImageName the first column
features_df = features_df[cols]
features_df.fillna(0, inplace=True)  # fill NaN with 0

# Remove features that don't appear in at least 10% of images AND at least once in each category
overall_threshold = 0.10 * len(features_df)
nature_df = features_df[features_df['Category'] == 'Nature']
urban_df = features_df[features_df['Category'] == 'Urban']

# Get feature columns (exclude ImageName and Category)
feature_cols = [col for col in features_df.columns if col not in ['ImageName', 'Category']]

# Filter features that meet both conditions: 10% overall presence AND at least once in each category
retained_features = []
for col in feature_cols:
    overall_present = (features_df[col] != 0).sum()
    nature_present = (nature_df[col] != 0).sum()
    urban_present = (urban_df[col] != 0).sum()
    if overall_present >= overall_threshold and nature_present >= 1 and urban_present >= 1:
        retained_features.append(col)

# Keep only retained features plus ImageName and Category
features_df = features_df[['ImageName', 'Category'] + retained_features]

# Combine with visual features
visual_features_df_all = pd.merge(visual_features_df, features_df, on='ImageName')
print(visual_features_df_all.head())
visual_features_df_all.to_csv('./stimuli/visual_features_extracted.csv')

# Example
# example_image = cv2.imread('./stimuli/nature/MDS140.jpg')
# gray = cv2.cvtColor(example_image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 100, 400)
# plt.figure(figsize=(10, 5))
# plt.imshow(edges, cmap='gray')
# plt.axis('off')
# plt.savefig('./figures/Canny_Edges_Example.png', dpi=600)
# plt.show()

# # run correlation with original visual features
# nature_merged = pd.merge(nature_features_df, stimuli_info, on='ImageName')
# features_cur = ['Hue_x', 'Bright_x', 'Saturaton_x', 'SDhue_x', 'SDsat_x', 'Sdbright_x', 'Entropy_x', 'SED_x', 'NSED_x', 'total_ED_x']
# features_orig = ['Hue_y', 'Bright_y', 'Saturaton_y', 'SDhue_y', 'SDsat_y', 'Sdbright_y', 'Entropy_y', 'SED_y', 'NSED_y', 'total_ED_y']
# correlations = {}
# for f_cur, f_orig in zip(features_cur, features_orig):
#     corr = nature_merged[f_cur].corr(nature_merged[f_orig])
#     correlations[f_cur] = corr
# print('Correlations between extracted and original features for nature stimuli:')
# for feature, corr in correlations.items():
#     print(f'{feature}: {corr:.4f}')
#
# mds_nature = nature_merged[nature_features_df['ImageName'].str.contains('Urban')]
# corr = {}
# for f_cur, f_orig in zip(features_cur, features_orig):
#     corr_val = mds_nature[f_cur].corr(mds_nature[f_orig])
#     corr[f_cur] = corr_val
# print('Correlations between extracted and original features for MDS nature stimuli:')
# for feature, corr_val in corr.items():
#     print(f'{feature}: {corr_val:.4f}')
#
# sample_image = list(nature_images.values())[0]
# gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
# x = cv2.Canny(gray, 100, 200)

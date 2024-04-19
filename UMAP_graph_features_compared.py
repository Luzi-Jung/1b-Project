import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap

# Read the CSV file and drop NaN values
data_frame = pd.read_csv('your/file/loaction/cleaned_data.csv')
data_frame = data_frame.dropna()

# Define the groups
reference_groups = ['M0', 'M1', 'M2']
groups_wanted = ['Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2']
all_groups = groups_wanted + reference_groups
subset_df = data_frame[data_frame['group_name'].isin(all_groups)]
subset_df['group_name'] = pd.Categorical(subset_df['group_name'], categories=all_groups, ordered=True)

# Select and prepare feature data
features_wanted = ['centers_area', 'centers_distance', 'cytoplasm_area', 'GaborMax1_Actin', 'GaborMin1_Actin', 'HarConCellAct', 'HarConCytoTub', 'HarConMembAct', 'HarCorrCellAct', 'HarCorrCytoTub', 'HarCorrMembAct', 'HarHomCellAct', 'HarHomCytoTub', 'HarHomMembAct', 'HarSVCellAct', 'HarSVCytoTub', 'HarSVMembAct', 'logNucbyRingActin', 'logNucbyRingTubulin', 'mean_prlength', 'MembranebyCytoOnlyActin', 'MembranebyCytoOnlyTubulin', 'NucbyCytoArea', 'NucbyRingActin', 'NucbyRingTubulin', 'NucPlusRingActin', 'NucPlusRingTubulin', 'RingbyCytoActin', 'RingbyCytoTubulin', 'ringregion_area']
subset_df = subset_df[['group_name'] + features_wanted]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(subset_df[features_wanted])

# Compute UMAP embedding
reducer = umap.UMAP(n_neighbors=8, min_dist=0, n_components=2)
embedding = reducer.fit_transform(X_scaled)
# Append UMAP coordinates to the dataframe
subset_df['UMAP_1'], subset_df['UMAP_2'] = embedding[:, 0], embedding[:, 1]

# k-means clustering
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
subset_df['cluster'] = kmeans.fit_predict(X_scaled)

# Define colors and markers (all filled for compatibility)
feature_colors = sns.color_palette("hsv", len(features_wanted))
group_markers = {group: marker for group, marker in zip(all_groups, 
                  ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', 'h', 'H', '8', '*', 'X', 'd', 'D'])}
color_dict = dict(zip(all_groups, sns.color_palette("husl", len(all_groups))))

# Function to create multiple plots
def create_plots(groups_subset, fig_title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, group in enumerate(groups_subset):
        plot_data = subset_df[subset_df['group_name'].isin([group] + reference_groups)]
        ax = sns.scatterplot(ax=axes[i], x='UMAP_1', y='UMAP_2', 
                             hue='group_name', style='group_name',
                             markers=group_markers, palette=color_dict,
                             data=plot_data)
        axes[i].set_title(f'Comparison of {group} with M0, M1, M2')
        if ax.legend_:
            ax.legend_.remove()
    plt.tight_layout()
    plt.suptitle(fig_title)
    plt.subplots_adjust(top=0.88)
    plt.show()

# Create two separate figures for different groups
create_plots(['Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2'], 'UMAP Comparisons: Cardiac to Card_M2')
create_plots(['SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2'], 'UMAP Comparisons: SIS_M0 to UBM_M2')
# Legend for features and groups in a separate figure
fig_leg, ax_leg = plt.subplots(figsize=(15, 2))
for feature, color in zip(features_wanted, feature_colors):
    ax_leg.scatter([], [], color=color, label=feature)
for group, marker in group_markers.items():
    ax_leg.scatter([], [], marker=marker, color='black', label=group)
ax_leg.legend(ncol=4, loc="center", frameon=False)
ax_leg.axis('off')
plt.show()

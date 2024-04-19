import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import umap.plot
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics.cluster import silhouette_samples

# Read the CSV file, drop NaN values
data_frame = pd.read_csv('/your/file/location/cleaned_data.csv')
data_frame = data_frame.dropna()

# Define the groups
groups_wanted = ['M0', 'M1', 'M2', 'Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2']
subset_df = data_frame[data_frame['group_name'].isin(groups_wanted)].copy()
#convert group_name column to categorical variable
subset_df.loc[:, 'group_name']= pd.Categorical(subset_df['group_name'], categories=groups_wanted, ordered=True)

features_wanted = ['group_name', 'CellArea', 'cellbody_area', 'Cell_Elongation', 'cell_full_length', 'cell_half_width', 'Cell_length_by_area', 'Cell_width_by_area', 'NucleusArea', 'Nuc_Elongation', 'Nuc_full_length', 'Nuc_half_width', 'Nuc_Roundness', 'roundness', 'body_roundness', 'percentProtrusion', 'protrusion_extent', 'total_protrusionarea', 'mean_protrusionarea', 'number_protrusions', 'skeleton_area', 'skeleton_node_count', 'skeletonareapercent', 'cytointensityActin', 'cytointensityTubulin', 'CytoIntensityH', 'CytoNonMembraneIntensityActin', 'CytoNonMembraneIntensityTubulin', 'MembraneIntensityActin', 'MembraneIntensityTubulin', 'NucIntensityActin', 'NucIntensityTubulin', 'NucIntensityH', 'ProtrusionIntensityActin', 'ProtrusionIntensityTubulin', 'ringIntensityActin', 'ringIntensityTubulin', 'RingIntensityH', 'WholeCellIntensityActin', 'WholeCellIntensityTubulin', 'WholeCellIntensityH', 'centers_area', 'centers_distance', 'cytoplasm_area', 'GaborMax1_Actin', 'GaborMin1_Actin', 'HarConCellAct', 'HarConCytoTub', 'HarConMembAct', 'HarCorrCellAct', 'HarCorrCytoTub', 'HarCorrMembAct', 'HarHomCellAct', 'HarHomCytoTub', 'HarHomMembAct', 'HarSVCellAct', 'HarSVCytoTub', 'HarSVMembAct', 'logNucbyRingActin', 'logNucbyRingTubulin', 'mean_prlength', 'MembranebyCytoOnlyActin', 'MembranebyCytoOnlyTubulin', 'NucbyCytoArea', 'NucbyRingActin', 'NucbyRingTubulin', 'NucPlusRingActin', 'NucPlusRingTubulin', 'RingbyCytoActin', 'RingbyCytoTubulin', 'ringregion_area','SERBrightCellAct', 'SERBrightCytoTub', 'SERBrightMembAct', 'SERBrightNuc', 'SERDarkCellAct', 'SERDarkCytoTub', 'SERDarkMembAct', 'SERDarkNuc', 'SEREdgeCellAct', 'SEREdgeCytoTub', 'SEREdgeMembAct', 'SEREdgeNuc', 'SERHoleCellAct', 'SERHoleCytoTub', 'SERHoleMembAct', 'SERHoleNuc', 'SERRidgeCellAct', 'SERRidgeCytoTub', 'SERRidgeMembAct', 'SERRidgeNuc', 'SERSaddleCellAct', 'SERSaddleCytoTub', 'SERSaddleMembAct', 'SERSaddleNuc', 'SERSpotCellAct', 'SERSpotCytoTub', 'SERSpotMembAct', 'SERSpotNuc', 'SERValleyCellAct', 'SERValleyCytoTub', 'SERValleyMembAct', 'SERValleyNuc']
subset_df = subset_df[features_wanted]

#extract numeric data, first column is group_name
data_X = subset_df.iloc[:, 1:].values

#scale data
scaler = StandardScaler()
X = scaler.fit_transform(data_X)

#kmeans
kmeans_param_grid = {'n_clusters': [4, 5, 8, 10]}
#generate all parameter options
grid=ParameterGrid(kmeans_param_grid)

# Manual parameter tuning
best_score = -np.inf
best_params = None

# Iterate over each parameter combo
for params in grid:
    reducer = umap.UMAP(min_dist=0, n_components=2, n_neighbors=8, random_state=84)
    embedding = reducer.fit_transform(X)
    # Perform clustering on embedding using KMeans
    clusterer = KMeans(**params)
    labels = clusterer.fit_predict(embedding)
    # Evaluate clustering using silhouette score
    score = silhouette_score(embedding, labels)

    # Check if current combo has best score
    if score > best_score:
        best_score = score
        best_params = params
        
print("Best Parameters are: ", best_params)
print("Best Silhouette Score: ", best_score)


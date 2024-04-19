import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datashader
import holoviews
import numba
import umap
import umap.plot
from colorcet import glasbey_bw
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans #pre-processing clustering
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import silhouette_samples


#Groups names for selecting are: 'M0', 'M1', 'M2', 'Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2'}

# Read the CSV file, drop NaN values
data_frame = pd.read_csv('your/file/location/normalised_macs_data.csv')
data_frame = data_frame.dropna()

#subset the dataframe to include all groups
groups_wanted = ['M0', 'M1', 'M2', 'Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2']
subset_df = data_frame[data_frame['group_name'].isin(groups_wanted)]
#subset data frame to include features wanted
features_wanted = ['CellArea', 'cellbody_area', 'Cell_Elongation', 'cell_full_length', 'cell_half_width', 'Cell_length_by_area', 'Cell_width_by_area', 'NucleusArea', 'Nuc_Elongation', 'Nuc_full_length', 'Nuc_half_width', 'Nuc_Roundness', 'roundness', 'body_roundness', 'percentProtrusion', 'protrusion_extent', 'total_protrusionarea', 'mean_protrusionarea', 'number_protrusions', 'skeleton_area', 'skeleton_node_count', 'skeletonareapercent', 'cytointensityActin', 'cytointensityTubulin', 'CytoIntensityH', 'CytoNonMembraneIntensityActin', 'CytoNonMembraneIntensityTubulin', 'MembraneIntensityActin', 'MembraneIntensityTubulin', 'NucIntensityActin', 'NucIntensityTubulin', 'NucIntensityH', 'ProtrusionIntensityActin', 'ProtrusionIntensityTubulin', 'ringIntensityActin', 'ringIntensityTubulin', 'RingIntensityH', 'WholeCellIntensityActin', 'WholeCellIntensityTubulin', 'WholeCellIntensityH', 'centers_area', 'centers_distance', 'cytoplasm_area', 'GaborMax1_Actin', 'GaborMin1_Actin', 'HarConCellAct', 'HarConCytoTub', 'HarConMembAct', 'HarCorrCellAct', 'HarCorrCytoTub', 'HarCorrMembAct', 'HarHomCellAct', 'HarHomCytoTub', 'HarHomMembAct', 'HarSVCellAct', 'HarSVCytoTub', 'HarSVMembAct', 'logNucbyRingActin', 'logNucbyRingTubulin', 'mean_prlength', 'MembranebyCytoOnlyActin', 'MembranebyCytoOnlyTubulin', 'NucbyCytoArea', 'NucbyRingActin', 'NucbyRingTubulin', 'NucPlusRingActin', 'NucPlusRingTubulin', 'RingbyCytoActin', 'RingbyCytoTubulin', 'ringregion_area','SERBrightCellAct', 'SERBrightCytoTub', 'SERBrightMembAct', 'SERBrightNuc', 'SERDarkCellAct', 'SERDarkCytoTub', 'SERDarkMembAct', 'SERDarkNuc', 'SEREdgeCellAct', 'SEREdgeCytoTub', 'SEREdgeMembAct', 'SEREdgeNuc', 'SERHoleCellAct', 'SERHoleCytoTub', 'SERHoleMembAct', 'SERHoleNuc', 'SERRidgeCellAct', 'SERRidgeCytoTub', 'SERRidgeMembAct', 'SERRidgeNuc', 'SERSaddleCellAct', 'SERSaddleCytoTub', 'SERSaddleMembAct', 'SERSaddleNuc', 'SERSpotCellAct', 'SERSpotCytoTub', 'SERSpotMembAct', 'SERSpotNuc', 'SERValleyCellAct', 'SERValleyCytoTub', 'SERValleyMembAct', 'SERValleyNuc']
subset_df = subset_df[features_wanted]

#extract numeric data, first column is group_name
data_X = subset_df.iloc[:, 1:].values

#scale data
scaler = StandardScaler()
X = scaler.fit_transform(data_X)

#UMAP embedding
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(X)

#umap
umap_param_grid = {
    'n_neighbors': [5, 8, 11, 15],
    'min_dist': [0, 0.5, 1],
    'n_components': [2]  
}
#generate all parameter options
grid=ParameterGrid(umap_param_grid)

#manual parameter tuning
best_score = -np.inf
best_params = None

#iterate over each parameter combo
for params in grid:
    reducer = umap.UMAP(**params)
    embedding = reducer.fit_transform(X)
    #perform clustering on embedding using KMeans
    clusterer = KMeans(n_clusters=6, n_init=10, random_state = 84)#set KMeans with explicit n_init, Kmeans ran 10 times to find optimum
    labels = clusterer.fit_predict(embedding)
    #evaluate embedding using silhouette score
    score = silhouette_score(embedding, labels)

    #check if current combo has best score
    if score > best_score:
        best_score = score
        best_params = params
        
print("Best Parameters are: ", best_params)
print("Best Silhouette Score: ", best_score)

##OUTPUT##
#Best Parameters are:  {'min_dist': 0, 'n_components': 2, 'n_neighbors': 11}
#Best Silhouette Score:  0.4524649


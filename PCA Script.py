#Install all packages needed
import numpy as np
import scipy
import sklearn
import numba
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from scipy import stats

#############################

#PCA analysis and scree plot code

# Read the CSV file, drop NaN values
data_frame = pd.read_csv('/Users/luzijungmayr/Desktop/Stuff/University Msc/Research Project 1b/Project/Data and Script/normalised_macs_data.csv')
data_frame = data_frame.dropna()

#Subset data frame to choose group wanted for PCA, M0 used as example
groups_wanted = ['M0', 'M1', 'M2', 'Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2']
subset_df = data_frame[data_frame['group_name'].isin(groups_wanted)].copy()
#convert group_name column to categorical variable
subset_df.loc[:, 'group_name']= pd.Categorical(subset_df['group_name'], categories=groups_wanted, ordered=True)

#extract numeric data, first column is group_name
data_X = subset_df.iloc[:, 1:].values

#scale data
scaler = StandardScaler()
X = scaler.fit_transform(data_X)

#PCA

#perform PCA
pca = PCA(n_components=102)
pc = pca.fit_transform(X)

#create new dataframe with PCs and add group info back to the dataframe
pca_df = pd.DataFrame(data=pc, columns=[f'PC{i}' for i in range(1, 103)]) #['PC1', 'PC2'])
pca_df['group'] = subset_df['group_name'].values

#plot PCA graph
fig = px.scatter(pca_df, x='PC1', y='PC2', color='group', title=f'PCA Plot for {groups_wanted} Data')
fig.show()

##REMOVE OUTLIERS FROM DATA##

# Calculate z-scores for each feature
z_scores = stats.zscore(X)

# Define a threshold for outlier detection
threshold = 4

# Find indices of outliers
outlier_indices = np.where(np.abs(z_scores) > threshold)
# Create a list to store removed data points
removed_data_points = []

# Iterate over outlier indices and collect corresponding data points
for idx in zip(*outlier_indices):
    data_point = {}
    data_point['Index'] = idx[0]  # Assuming the first column is the index
    for i, val in enumerate(data_X[idx[0]]):
        data_point[f'Feature_{i+1}'] = val
    removed_data_points.append(data_point)

# Convert the list to a DataFrame for better visualization
removed_df = pd.DataFrame(removed_data_points)

# Display the DataFrame
print("Removed Data Points:")
print(removed_df)

#export to csv file
removed_df.to_csv('removed_data_frame.csv')

# Step 2: Remove outliers
# Remove entire rows (data points) containing outliers
cleaned_data_X = np.delete(X, outlier_indices[0], axis=0)

# You may also want to remove corresponding rows from subset_df if necessary
cleaned_subset_df = subset_df.drop(outlier_indices[0])

# Now you can proceed with PCA using the cleaned data
# Scale cleaned data
cleaned_X = scaler.fit_transform(cleaned_data_X)

# Perform PCA on cleaned data
pca_cleaned = PCA(n_components=102)
pc_cleaned = pca_cleaned.fit_transform(cleaned_X)

# Create new dataframe with PCs and add group info back to the dataframe
pca_cleaned_df = pd.DataFrame(data=pc_cleaned, columns=[f'PC{i}' for i in range(1, 103)])
pca_cleaned_df['group'] = cleaned_subset_df['group_name'].values

# Plot PCA graph for cleaned data
fig_cleaned = px.scatter(pca_cleaned_df, x='PC1', y='PC2', color='group', title=f'PCA Plot for {groups_wanted} Data (after outlier removal)')
fig_cleaned.show()


# Convert cleaned df to pd
cleaned_df = pd.DataFrame(cleaned_subset_df)
#save cleaned df to csv
cleaned_df.to_csv('cleaned_data.csv')








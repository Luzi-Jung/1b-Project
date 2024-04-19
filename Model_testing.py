import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from umap import UMAP
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import zip_longest

# Read the CSV file, drop NaN values
data_frame = pd.read_csv('your/file/location/cleaned_data.csv', index_col=0)
data_frame = data_frame.dropna()

# Define parameters and execute the functions
reference_groups = ['M0', 'M1', 'M2']
tested_groups = ['Cardiac', 'SIS', 'UMB', 'Card_M0', 'Card_M1', 'Card_M2', 'SIS_M0', 'SIS_M1', 'SIS_M2', 'UBM_M0', 'UBM_M1', 'UBM_M2']
feature_columns = ['SERBrightCellAct', 'SERBrightCytoTub', 'SERBrightMembAct', 'SERBrightNuc', 'SERDarkCellAct', 'SERDarkCytoTub', 'SERDarkMembAct', 'SERDarkNuc', 'SEREdgeCellAct', 'SEREdgeCytoTub', 'SEREdgeMembAct', 'SEREdgeNuc', 'SERHoleCellAct', 'SERHoleCytoTub', 'SERHoleMembAct', 'SERHoleNuc', 'SERRidgeCellAct', 'SERRidgeCytoTub', 'SERRidgeMembAct', 'SERRidgeNuc', 'SERSaddleCellAct', 'SERSaddleCytoTub', 'SERSaddleMembAct', 'SERSaddleNuc', 'SERSpotCellAct', 'SERSpotCytoTub', 'SERSpotMembAct', 'SERSpotNuc', 'SERValleyCellAct', 'SERValleyCytoTub', 'SERValleyMembAct', 'SERValleyNuc']
umap_params = {'min_dist': 0, 'n_components': 2, 'n_neighbors': 8}
kmeans_clusters = 4

# Define functions
def analyze_group(data, feature_columns, umap_params, kmeans_clusters):
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)

    umap_embedding = UMAP(**umap_params).fit(train_data[feature_columns])
    train_embedding = umap_embedding.transform(train_data[feature_columns])
    test_embedding = umap_embedding.transform(test_data[feature_columns])

    kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=42)
    train_labels = kmeans.fit_predict(train_embedding)
    test_labels = kmeans.predict(test_embedding)

    silhouette_train = silhouette_score(train_embedding, train_labels)
    davies_bouldin_train = davies_bouldin_score(train_embedding, train_labels)
    silhouette_test = silhouette_score(test_embedding, test_labels)
    davies_bouldin_test = davies_bouldin_score(test_embedding, test_labels)

    return silhouette_train, davies_bouldin_train, silhouette_test, davies_bouldin_test

def analyze_reference_groups(data_frame, reference_groups, feature_columns, umap_params, kmeans_clusters):
    results = []

    for reference_group in reference_groups:
        silhouette_train, davies_bouldin_train, silhouette_test, davies_bouldin_test = analyze_group(data_frame[data_frame['group_name'] == reference_group], feature_columns, umap_params, kmeans_clusters)
        results.append([silhouette_train, davies_bouldin_train, silhouette_test, davies_bouldin_test])

    return pd.DataFrame(results, columns=['Silhouette Train', 'Davies-Bouldin Train', 'Silhouette Test', 'Davies-Bouldin Test'], index=reference_groups)

def compare_to_group(data_frame, reference_group, test_group, feature_columns, umap_params, kmeans_clusters):
    reference_data = data_frame[data_frame['group_name'] == reference_group]

    umap_embedding = UMAP(**umap_params).fit(reference_data[feature_columns])
    reference_embedding = umap_embedding.transform(reference_data[feature_columns])
    test_data = data_frame[data_frame['group_name'] == test_group]
    test_embedding = umap_embedding.transform(test_data[feature_columns])

    kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=42)
    reference_labels = kmeans.fit_predict(reference_embedding)
    test_labels = kmeans.predict(test_embedding)

    silhouette_reference = silhouette_score(reference_embedding, reference_labels)
    silhouette_test = silhouette_score(test_embedding, test_labels)
    
    davies_bouldin_reference = davies_bouldin_score(reference_embedding, reference_labels)
    davies_bouldin_test = davies_bouldin_score(test_embedding, test_labels)

    return silhouette_reference, davies_bouldin_reference, silhouette_test, davies_bouldin_test

def compare_to_reference_groups(data_frame, reference_groups, tested_groups, feature_columns, umap_params, kmeans_clusters):
    results = []

    for test_group in tested_groups:
        scores = []
        for reference_group in reference_groups:
            silhouette_train, davies_bouldin_train, silhouette_test, davies_bouldin_test = compare_to_group(data_frame, reference_group, test_group, feature_columns, umap_params, kmeans_clusters)
            scores.extend([silhouette_train, silhouette_test, davies_bouldin_train, davies_bouldin_test])
        results.append([test_group] + scores)

    return pd.DataFrame(results, columns=['Group'] + [f'{ref}_Silhouette_Train' for ref in reference_groups] + [f'{ref}_Silhouette_Test' for ref in reference_groups] + [f'{ref}_Davies-Bouldin_Train' for ref in reference_groups] + [f'{ref}_Davies-Bouldin_Test' for ref in reference_groups])

# Analyze reference groups for silhouette scores
reference_group_results_silhouette = analyze_reference_groups(data_frame, reference_groups, feature_columns, umap_params, kmeans_clusters)

# Compare tested groups to reference groups for silhouette scores
tested_group_results_silhouette = compare_to_reference_groups(data_frame, reference_groups, tested_groups, feature_columns, umap_params, kmeans_clusters)

# Print reference group results with silhouette scores
print("Reference Group Results with Silhouette Scores:")
print(reference_group_results_silhouette[['Silhouette Train', 'Silhouette Test']])

# Print tested group results with silhouette scores
print("\nTested Group Results with Silhouette Scores:")
print(tested_group_results_silhouette[[col for col in tested_group_results_silhouette.columns if 'Silhouette' in col]])

# Analyze reference groups for Davies-Bouldin scores
reference_group_results_davies_bouldin = analyze_reference_groups(data_frame, reference_groups, feature_columns, umap_params, kmeans_clusters)

# Compare tested groups to reference groups for Davies-Bouldin scores
tested_group_results_davies_bouldin = compare_to_reference_groups(data_frame, reference_groups, tested_groups, feature_columns, umap_params, kmeans_clusters)

# Print reference group results with Davies-Bouldin scores
print("\nReference Group Results with Davies-Bouldin Scores:")
print(reference_group_results_davies_bouldin[['Davies-Bouldin Train', 'Davies-Bouldin Test']])

# Print tested group results with Davies-Bouldin scores
print("\nTested Group Results with Davies-Bouldin Scores:")
print(tested_group_results_davies_bouldin[[col for col in tested_group_results_davies_bouldin.columns if 'Davies-Bouldin' in col]])

# Save the tested group results to a CSV file
reference_group_results_silhouette.to_csv('reference_group_results_silhouette_SER.csv')
tested_group_results_silhouette.to_csv('tested_group_results_silhouette_SER.csv')

reference_group_results_davies_bouldin.to_csv('reference_group_results_davies_bouldin_SER.csv')
tested_group_results_davies_bouldin.to_csv('tested_group_results_davies_bouldin_SER.csv')


# Comapre features distribution
def compare_feature_coordinates(data_frame, reference_group, test_group, feature_columns, umap_params, kmeans_clusters):
    reference_data = data_frame[data_frame['group_name'] == reference_group]
    test_data = data_frame[data_frame['group_name'] == test_group]

    results = []

    for reference_column in feature_columns:
        reference_subset = reference_data[[reference_column]]
        test_subset = test_data.sample(n=len(reference_data), random_state=42)[[reference_column]]  # Subsample test group

        umap_embedding = UMAP(**umap_params).fit(reference_subset)
        reference_embedding = umap_embedding.transform(reference_subset)
        test_embedding = umap_embedding.transform(test_subset)

        kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=42)
        reference_labels = kmeans.fit_predict(reference_embedding)
        test_labels = kmeans.predict(test_embedding)

        for i in range(len(test_embedding)):
            results.append({
                'Tested Group': test_group,
                'Feature': reference_column,
                'Cluster Label': test_labels[i],
                'Coordinates': test_embedding[i]
            })

    return pd.DataFrame(results)

def calculate_feature_similarity(data_frame, reference_group, feature_columns, umap_params, kmeans_clusters):
    reference_data = data_frame[data_frame['group_name'] == reference_group]
    feature_similarity_scores = []

    for feature_column in feature_columns:
        # Selecting the current feature column
        feature_subset = reference_data[[feature_column]]

        # Applying UMAP for dimensionality reduction
        umap_embedding = UMAP(**umap_params).fit(feature_subset)
        feature_embedding = umap_embedding.transform(feature_subset)

        # Performing KMeans clustering
        kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(feature_embedding)

        # Calculating silhouette score
        silhouette = silhouette_score(feature_embedding, labels)
        feature_similarity_scores.append((feature_column, silhouette))

    return feature_similarity_scores

# Calculate feature similarity for each reference group
reference_group_feature_similarity = {}
for reference_group in reference_groups:
    feature_similarity_scores = calculate_feature_similarity(data_frame, reference_group, feature_columns, umap_params, kmeans_clusters)
    reference_group_feature_similarity[reference_group] = feature_similarity_scores

# Print feature similarity scores for each reference group
for reference_group, similarity_scores in reference_group_feature_similarity.items():
    print(f"Feature similarity scores for {reference_group}:")
    for feature, similarity in similarity_scores:
        print(f"{feature}: {similarity}")
    print("\n")

def calculate_feature_similarity_all_groups(data_frame, reference_groups, tested_groups, feature_columns, umap_params, kmeans_clusters):
    similarity_scores_all_groups = []

    for test_group in tested_groups:
        for feature_column in feature_columns:
            for reference_group in reference_groups:
                silhouette_score = compare_to_group(data_frame, reference_group, test_group, [feature_column], umap_params, kmeans_clusters)[2]
                similarity_scores_all_groups.append({
                    'Tested Group': test_group,
                    'Feature Column': feature_column,
                    'Reference Group': reference_group,
                    'Silhouette Score': silhouette_score
                })

    return pd.DataFrame(similarity_scores_all_groups)

# Calculate silhouette scores for all tested groups and feature columns compared to each reference group
df_similarity_scores = calculate_feature_similarity_all_groups(data_frame, reference_groups, tested_groups, feature_columns, umap_params, kmeans_clusters)

# Print or manipulate the DataFrame as needed
print(df_similarity_scores)
df_similarity_scores.to_csv('df_similarity_scores_SER.csv')

# Define a function to create heatmaps for each tested group
def create_heatmap(tested_group):
    # Filter the DataFrame for the specified tested group
    df_tested_group = df_similarity_scores[df_similarity_scores['Tested Group'] == tested_group]
    
    # Create a pivot table with Reference Group and Feature Column as indices
    pivot_table = df_tested_group.pivot(index='Feature Column', columns='Reference Group', values='Silhouette Score')
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='inferno', annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Silhouette Score'})
    plt.title(f'Silhouette Scores for {tested_group}', fontsize=16)
    plt.xlabel('Reference Group', fontsize=12)
    plt.ylabel('Feature Column', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Create heatmaps for each tested group
for tested_group in tested_groups:
    create_heatmap(tested_group)


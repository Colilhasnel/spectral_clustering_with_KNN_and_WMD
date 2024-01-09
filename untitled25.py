# -*- coding: utf-8 -*-
"""Untitled25.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vMtDZMuthnceHkfIkq3aDN5GjDaxMmQ5
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def calculate_mahalanobis_distances(data, mean_vector, inv_cov_matrix):
    centered_data = data - mean_vector
    try:
        mahalanobis_distances = np.sqrt(np.sum(centered_data @ inv_cov_matrix * centered_data, axis=1))
        return mahalanobis_distances
    except Exception as e:
        print(f"Error in Mahalanobis distance calculation: {e}")
        return np.full(len(data), np.nan)

def calculate_similarity_matrix(data, k, scale_parameter, knn_method='method_B'):
    mahalanobis_distances = calculate_mahalanobis_distances(data, np.mean(data, axis=0), np.linalg.inv(np.cov(data, rowvar=False)))

    if knn_method == 'method_A':
        k_nearest_neighbors = np.argsort(mahalanobis_distances)[:, :k]
        is_neighbor = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            is_neighbor[i, k_nearest_neighbors[i]] = 1
    elif knn_method == 'method_B':
        is_neighbor = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            k_nearest_neighbors = np.argsort(mahalanobis_distances[i])[:k]
            is_neighbor[i, k_nearest_neighbors] = 1

    similarities = np.exp(-0.5 * np.square(mahalanobis_distances / scale_parameter)) * is_neighbor
    return similarities


def MDLSC_algorithm(data, num_clusters, k_neighbors=5, scale_parameter=1.0, knn_method='method_A'):
    try:
        # Calculate the similarity matrix
        similarity_matrix = calculate_similarity_matrix(data, k_neighbors, scale_parameter, knn_method)

        # Calculate the degree matrix
        degree_matrix = np.diag(similarity_matrix.sum(axis=1))

        # Calculate Laplacian matrix
        laplacian_matrix = degree_matrix - similarity_matrix

        # Regularize the Laplacian matrix
        regularization_term = 1e-5
        regularized_laplacian_matrix = laplacian_matrix + regularization_term * np.eye(laplacian_matrix.shape[0])

        # Calculate embedding using the first k smallest eigenvectors
        embedding = np.linalg.eigh(regularized_laplacian_matrix)[1][:, :num_clusters]

        # Normalize the embedding
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)

        # Apply k-means clustering on the normalized embedding
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, tol=1e-1, random_state=42)
        clusters = kmeans.fit_predict(embedding)

        # Print unique cluster values
        print("Unique Cluster Values:", np.unique(clusters))

        # Ensure the input data is 2D for silhouette score calculation
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, clusters)
        print(f"Silhouette Score: {silhouette_avg}")

        # Assuming subset_data_std is a NumPy array and subset_data is the original DataFrame
        print("Subset of Standardized Data (First 5 Rows with Column Names):")
        subset_data_head = data[:5, :]  # Use the original data instead of subset_data_std
        print(subset_data_head)

        # Visualize the results
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=50)
        plt.title('Spectral Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        clusters = np.zeros(len(data), dtype=int)  # Assign all points to a single cluster in case of an error

    return clusters

# # Load your dataset
# data = pd.read_csv('/content/drive/MyDrive/newdataset_spectral_clustering.dat', delimiter='\s+')

# # Remove leading and trailing whitespaces from column names
# data.columns = data.columns.str.strip()

# # Handling missing values (replace NaN with 0)
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data.dropna(inplace=True)
# data_filled = data.fillna(0)

# # Handling missing values (replace NaN with 0)
# data_filled = np.nan_to_num(data, nan=0)

# Load your dataset
data = pd.read_csv('/content/drive/MyDrive/newdataset_spectral_clustering.dat', delimiter='\s+')

# Remove leading and trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Handling missing values (replace NaN with mean)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Remove rows with NaN values
data = data.dropna()

#Specify the columns to be used in clustering
selected_columns = ['AOD440', 'AOD500', 'AOD675', 'AOD870', 'AOD1020', 'AOD_F_440', 'AOD_F_500',
                    'AOD_C_440', 'AOD_C_500', 'SSA_440', 'SSA_500', 'SSA_675', 'SSA_870', 'SSA_1020',
                    'ASY_440', 'ASY_500', 'ASY675', 'ASY_870', 'ASY_1020', 'EAE440-870', 'AAOD500',
                    'AAE-440-870', 'FMF500']
# Specify the columns to be used in clustering
selected_columns = data.columns.tolist()

# Convert to floating-point numbers
subset_data = data[selected_columns].astype(float)

# Data Standardization
scaler = StandardScaler()
subset_data_std = scaler.fit_transform(subset_data)

# Replace NaN with 0
subset_data_std = np.nan_to_num(subset_data_std)

# Number of clusters
num_clusters = 3

# Apply MDLSC algorithm with tuned parameters and method_A for knn_method
clusters = MDLSC_algorithm(subset_data_std, num_clusters, k_neighbors=1, scale_parameter=2.0, knn_method='method_B')

# Print unique cluster values
print(np.unique(clusters))

# Assuming subset_data_std is a NumPy array and subset_data is the original DataFrame
print("Subset of Standardized Data (First 5 Rows with Column Names):")
subset_data_head = pd.DataFrame(subset_data_std[:5, :], columns=selected_columns)
print(subset_data_head)

# Visualize the results
plt.scatter(subset_data_std[:, 0], subset_data_std[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('Spectral Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# # Specify the columns to be used in clustering
# selected_columns = ['AOD440', 'AOD500', 'AOD675', 'AOD870', 'AOD1020', 'AOD_F_440', 'AOD_F_500',
#                     'AOD_C_440', 'AOD_C_500', 'SSA_440', 'SSA_500', 'SSA_675', 'SSA_870', 'SSA_1020',
#                     'ASY_440', 'ASY_500', 'ASY675', 'ASY_870', 'ASY_1020', 'EAE440-870', 'AAOD500',
#                     'AAE-440-870', 'FMF500']

# # Data Standardization
# scaler = StandardScaler()
# subset_data_std = scaler.fit_transform(subset_data)

# # Number of clusters
# num_clusters = 3

# # Apply MDLSC algorithm with tuned parameters and method_A for knn_method
# clusters = MDLSC_algorithm(subset_data_std, num_clusters, k_neighbors=5, scale_parameter=2.0, knn_method='method_B')

# # Print unique cluster values
# print(np.unique(clusters))

# # Specify the columns to be used in clustering
# selected_columns = data.columns.tolist()

# # Check if the number of selected columns matches the number of columns in subset_data_std
# if len(selected_columns) != subset_data_std.shape[1]:
#     raise ValueError("Number of selected columns does not match the number of columns in the standardized data.")

# # Assuming subset_data_std is a NumPy array and subset_data is the original DataFrame
# print("Subset of Standardized Data (First 5 Rows with Column Names):")
# subset_data_head = pd.DataFrame(subset_data_std[:5, :], columns=selected_columns)
# print(subset_data_head)



# # Visualize the results
# plt.scatter(subset_data_std[:, 0], subset_data_std[:, 1], c=clusters, cmap='viridis', s=50)
# plt.title('Spectral Clustering Results')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

# Define the output directory
output_directory = '/content/Leh/'
os.makedirs(output_directory, exist_ok=True)

# Create a larger figure
plt.figure(figsize=(20, 15))

# Scatter plot for SSA_440 vs EAE440-870
plt.subplot(2, 2, 1)
scatter = plt.scatter(subset_data_std[:, 19], subset_data_std[:, 0], c=clusters, cmap='viridis', alpha=0.7)
plt.title('AOD440 vs EAE440-870')
plt.xlabel('EAE440-870')
plt.ylabel('AOD440')
plt.colorbar(scatter)

# Scatter plot for SSA_440 vs EAE440-870
plt.subplot(2, 2, 2)
scatter = plt.scatter(subset_data_std[:, 19], subset_data_std[:, 9], c=clusters, cmap='viridis', alpha=0.7)
plt.title('SSA_440 vs EAE440-870')
plt.xlabel('EAE440-870')
plt.ylabel('SSA_440')
plt.colorbar(scatter)

# Scatter plot for AOD_F_440 vs EAE440-870
plt.subplot(2, 2, 3)
scatter = plt.scatter(subset_data_std[:, 19], subset_data_std[:, 5], c=clusters, cmap='viridis', alpha=0.7)
plt.title('AOD_F_440 vs EAE440-870')
plt.xlabel('EAE440-870')
plt.ylabel('AOD_F_440')
plt.colorbar(scatter)

# Scatter plot for AOD500 vs EAE440-870
plt.subplot(2, 2, 4)
scatter = plt.scatter(subset_data_std[:, 19], subset_data_std[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('AOD500 vs EAE440-870')
plt.xlabel('EAE440-870')
plt.ylabel('AOD500')
plt.colorbar(scatter)

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(output_directory, 'assigned_clusters_aod500_eae440_subplot.jpg'))

# Show the plot
plt.show()
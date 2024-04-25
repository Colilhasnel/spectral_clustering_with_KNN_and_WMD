import os
import numpy as np
import pandas as pd
import math
import get_calculated_data as data_storred
from sklearn.cluster import KMeans


DATA = pd.read_csv("newDATAset_spectral_clustering.dat", delimiter="\s+")

DATA.columns = DATA.columns.str.strip()
DATA.replace([np.inf, -np.inf], np.nan, inplace=True)
DATA.dropna(inplace=True)

DATA = DATA.dropna()

DATA.reset_index(inplace=True)

selected_columns = [
    "AOD440",
    "AOD500",
    "AOD675",
    "AOD870",
    "AOD1020",
    "AOD_F_440",
    "AOD_F_500",
    "AOD_C_440",
    "AOD_C_500",
    "SSA_440",
    "SSA_500",
    "SSA_675",
    "SSA_870",
    "SSA_1020",
    "ASY_440",
    "ASY_500",
    "ASY675",
    "ASY_870",
    "ASY_1020",
    "EAE440-870",
    "AAOD500",
    "AAE-440-870",
    "FMF500",
]

DATA = DATA[selected_columns]


N = len(DATA.axes[0])
P = len(DATA.axes[1])

MEAN = DATA.mean(axis=0)

STANDARD_DEVIATION = DATA.std()

VARIABLES = list(DATA.columns)

DATA_STATE = True


def calculate_corr_matrix(matrix_name, get_data):
    corr_matrix = []

    if get_data and data_storred.check_data(matrix_name):
        corr_matrix = data_storred.get_data(matrix_name)
    else:
        data = np.array(DATA)

        corr_matrix = np.corrcoef(data, rowvar=False)

        corr_matrix_data = pd.DataFrame(corr_matrix)
        corr_matrix_data.to_csv(os.path.join("calculated_data", matrix_name))

    return corr_matrix


def calculate_WMD(matrix_name, corr_matrix, get_data):
    WMD = np.zeros((N, N), dtype=np.float64)

    if get_data and data_storred.check_data(matrix_name):
        WMD = data_storred.get_data(matrix_name)
    else:
        for i in range(0, N):
            for j in range(i, N):
                X = np.array(DATA.loc[i])
                Y = np.array(DATA.loc[j])

                X = np.transpose(X)
                Y = np.transpose(Y)

                ans = np.dot(corr_matrix, X - Y)
                ans = np.dot(np.transpose(X - Y), ans)
                ans = math.sqrt(ans)

                WMD[i][j] = ans

        for i in range(0, N):
            for j in range(0, i):
                WMD[i][j] = WMD[j][i]

        WMD_data = pd.DataFrame(WMD)
        WMD_data.to_csv(os.path.join("calculated_data", matrix_name))

    return WMD


def calculate_similarity_matrix(matrix_name, WMD, get_data, k=3, z=3):
    similarity_matrix = np.zeros((N, N), dtype=np.float64)

    if get_data and data_storred.check_data(matrix_name):
        similarity_matrix = data_storred.get_data(matrix_name)
    else:
        k_nn = np.zeros((N, N), dtype=np.bool_)

        for i in range(0, N):
            neighbor_WMD = WMD[i].copy()
            neighbor_WMD[i] = math.inf

            for j in range(0, k):
                min_WMD = np.amin(neighbor_WMD)

                idx_neighbor = np.where(neighbor_WMD == min_WMD)[0][0]
                neighbor_WMD[idx_neighbor] = math.inf

                k_nn[i][idx_neighbor] = 1

        knn_data = pd.DataFrame(k_nn)
        knn_data.to_csv(os.path.join("calculated_data", matrix_name))

        for i in range(0, N):
            for j in range(i, N):
                if k_nn[i][j] and k_nn[j][i]:
                    coeff = (WMD[i][j] ** 2) / (2 * z)
                    similarity_matrix[i][j] = math.exp(-coeff)
                    similarity_matrix[j][i] = similarity_matrix[i][j]

        similarity_matrix_data = pd.DataFrame(similarity_matrix)

        similarity_matrix_data.to_csv(os.path.join("calculated_data", matrix_name))

    return similarity_matrix


def calculate_diagonal_matrix(matrix_name, W, get_data):
    D = []

    if get_data and data_storred.check_data(matrix_name):
        D = data_storred.get_data(matrix_name)
    else:
        D = np.diag(W.sum(axis=1))
        regularization_term = 1e-5

        D = D + np.eye(D.shape[0]) * regularization_term

        D_data = pd.DataFrame(D)
        D_data.to_csv(os.path.join("calculated_data", matrix_name))

    return D


def calculate_regularized_laplacian_matrix(matrix_name, W, D, get_data):
    regularized_L = []

    if get_data and data_storred.check_data(matrix_name):
        regularized_L = data_storred.get_data(matrix_name)
    else:
        laplaican_matrix_L = D - W

        D_inv_sqrt = np.linalg.inv(D ** (1 / 2))

        regularized_L = np.dot(laplaican_matrix_L, D_inv_sqrt)
        regularized_L = np.dot(D_inv_sqrt, regularized_L)

        regularization_term = 1e-5
        regularized_L = regularized_L + regularization_term * np.eye(
            regularized_L.shape[0]
        )

        regularized_L_data = pd.DataFrame(regularized_L)
        regularized_L_data.to_csv(os.path.join("calculated_data", matrix_name))

    return regularized_L


def calculate_k_smallest_normalized_eigenvectors(
    matrix_name, eigenvectors, k, get_data
):
    matrix_name = str(k) + "_" + matrix_name

    k_smallest_eig_V = []

    if get_data and data_storred.check_data(matrix_name):
        k_smallest_eig_V = data_storred.get_data(matrix_name)
    else:
        k_smallest_eig_V = eigenvectors[:, :k]

        k_smallest_eig_V /= np.linalg.norm(k_smallest_eig_V, axis=1, keepdims=True)

        np.nan_to_num(k_smallest_eig_V, copy=False, nan=(1 / np.sqrt(k)))

        k_smallest_eig_V_data = pd.DataFrame(k_smallest_eig_V)
        k_smallest_eig_V_data.to_csv(os.path.join("calculated_data", matrix_name))

    return k_smallest_eig_V


corr_matrix_R = calculate_corr_matrix("corr_matrix_R.csv", DATA_STATE)

WMD_matrix = calculate_WMD("WMD_matrix.csv", corr_matrix_R, DATA_STATE)

similarity_matrix_W = calculate_similarity_matrix(
    "similarity_matrix_W.csv", WMD_matrix, DATA_STATE, 3
)

diagonal_matrix_D = calculate_diagonal_matrix(
    "diagonal_matrix_D.csv", similarity_matrix_W, DATA_STATE
)

regularized_laplacian_matrix_L = calculate_regularized_laplacian_matrix(
    "regularized_laplacian_matrix_L.csv",
    similarity_matrix_W,
    diagonal_matrix_D,
    DATA_STATE,
)

eigenvalues_L, eigenvectors_L = np.linalg.eigh(regularized_laplacian_matrix_L)

eigenvectors_L_data = pd.DataFrame(eigenvectors_L)
eigenvectors_L_data.to_csv("calculated_data/eigenvectos_L.csv")

num_clusters = 3

eigenvectors_P = calculate_k_smallest_normalized_eigenvectors(
    "eigenvectors_P.csv", eigenvectors_L, num_clusters, DATA_STATE
)

X = eigenvectors_P

kmeans = KMeans(
    n_clusters=num_clusters, init="k-means++", random_state=2, max_iter=300, tol=1e-1
)
kmeans.fit(X)

pred = kmeans.labels_

cluster_prediction_data = pd.DataFrame(pred)
cluster_prediction_data.rename(columns={0: "Labels"}, inplace=True)
cluster_prediction_data.to_csv("calculated_data/cluster_prediction.csv")
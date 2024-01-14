import os
import numpy as np
import pandas as pd
import math
import get_calculated_data as data_storred

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


def calculate_corr_matrix(matrix_name):
    corr_matrix = []

    if data_storred.check_data(matrix_name):
        corr_matrix = data_storred.get_data(matrix_name)
    else:
        data = np.array(DATA)

        corr_matrix = np.corrcoef(data, rowvar=False)

        corr_matrix_data = pd.DataFrame(corr_matrix)
        corr_matrix_data.to_csv("calculated_data/corr_matrix_R.csv")

    return corr_matrix


def calculate_WMD(matrix_name, corr_matrix):
    WMD = np.zeros((N, N), dtype=np.float64)

    if data_storred.check_data(matrix_name):
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
        WMD_data.to_csv("calculated_data/WMD_matrix.csv")

    return WMD


def calculate_similarity_matrix(matrix_name, WMD, k=3, z=3):
    similarity_matrix = np.zeros((N, N), dtype=np.float64)

    if data_storred.check_data(matrix_name):
        similarity_matrix = data_storred.get_data(matrix_name)
    else:
        k_nn = np.zeros((N, N), dtype=np.bool_)

        for i in range(0, N):
            neighbor_WMD = WMD[i].copy()
            neighbor_WMD[i] = math.inf

            for j in range(0, k):
                min_WMD = min(neighbor_WMD)

                idx_neighbor = np.where(neighbor_WMD == min_WMD)[0][0]
                neighbor_WMD[idx_neighbor] = math.inf

                k_nn[i][idx_neighbor] = 1

        knn_data = pd.DataFrame(k_nn)
        knn_data.to_csv("calculated_data/k_nn_data.csv")

        for i in range(0, N):
            for j in range(i, N):
                if k_nn[i][j] and k_nn[j][i]:
                    coeff = (WMD[i][j] ** 2) / (2 * z)
                    similarity_matrix[i][j] = math.exp(-coeff)
                    similarity_matrix[j][i] = similarity_matrix[i][j]

        similarity_matrix_data = pd.DataFrame(similarity_matrix)

        similarity_matrix_data.to_csv("calculated_data/similarity_matrix.csv")

    return similarity_matrix


def calculate_diagonal_matrix(matrix_name, W):
    D = []

    if data_storred.check_data(matrix_name):
        D = data_storred.get_data(matrix_name)
    else:
        regularization_term = 1e-5

        D = np.diag(W.sum(axis=1))
        D = D + np.eye(D.shape[0]) * regularization_term

        D_data = pd.DataFrame(D)
        D_data.to_csv("calculated_data/diagonal_matrix.csv")

    return D


def calculate_regularized_laplacian_matrix(matrix_name, W, D):
    regularized_L = []

    if data_storred.check_data(matrix_name):
        regularized_L = data_storred.get_data(matrix_name)
    else:
        laplaican_matrix_L = D - W

        D_inv_sqrt = np.linalg.inv(D ** (1 / 2))

        regularized_L = np.dot(laplaican_matrix_L, D_inv_sqrt)
        regularized_L = np.dot(D_inv_sqrt, regularized_L)

        regularized_L_data = pd.DataFrame(regularized_L)
        regularized_L_data.to_csv("calculated_data/regularized_laplacian_matrix.csv")

    return regularized_L


def calculate_k_smallest_eigenvectors(matrix_name, eigenvalues, eigenvectors, k):
    matrix_name = str(k) + "_" + matrix_name

    k_smallest_eig_V = []

    if data_storred.check_data(matrix_name):
        k_smallest_eig_V = data_storred.get_data(matrix_name)
    else:
        indexes = []
        for i in range(0, k):
            min_element = min(eigenvalues)
            print(min_element)
            idx = np.where(eigenvalues == min_element)[0][0]
            indexes.append(idx)

            eigenvalues[idx] = math.inf
            k_smallest_eig_V.append(eigenvectors[:, idx])

        k_smallest_eig_V = np.array(k_smallest_eig_V, dtype=np.float64)
        k_smallest_eig_V = np.transpose(k_smallest_eig_V)

        eig_V_data = pd.DataFrame(k_smallest_eig_V)
        eig_V_data.to_csv("calculated_data/" + matrix_name)

    return k_smallest_eig_V


corr_matrix_R = calculate_corr_matrix("corr_matrix_R.csv")

WMD_matrix = calculate_WMD("WMD_matrix.csv", corr_matrix_R)

similarity_matrix_W = calculate_similarity_matrix(
    "similarity_matrix_W.csv", WMD_matrix, 3
)

diagonal_matrix_D = calculate_diagonal_matrix(
    "diagonal_matrix_D.csv", similarity_matrix_W
)

regularized_laplacian_matrix_L = calculate_regularized_laplacian_matrix(
    "regularized_laplacian_matrix_L.csv", similarity_matrix_W, diagonal_matrix_D
)

eigenvalues_L, eigenvectors_L = np.linalg.eigh(regularized_laplacian_matrix_L)

eigenvectors_P = calculate_k_smallest_eigenvectors(
    "eigenvector_matrix_P.csv", eigenvalues_L, eigenvectors_L, 3
)

print("Done")

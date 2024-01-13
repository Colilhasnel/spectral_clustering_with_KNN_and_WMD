import os
import numpy as np
import pandas as pd
import math

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


def calculate_corr_matrix():
    data = np.array(DATA)

    corr_matrix = np.corrcoef(data, rowvar=False)

    corr_matrix_data = pd.DataFrame(corr_matrix)
    corr_matrix_data.to_csv("calculated_data/corr_matrix_R.csv")

    return corr_matrix


def calculate_WMD(corr_matrix):
    WMD = np.zeros((N, N), dtype=np.float64)

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


def calculate_similarity_matrix(WMD, k=3, z=3):
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

    # for i in range(0, N):
    #     for j in range(i + 1, N):
    #         if k_nn[i][j] and k_nn[j][i]:
    #             print((i, j))

    similarity_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(0, N):
        for j in range(i, N):
            if k_nn[i][j] and k_nn[j][i]:
                coeff = (WMD[i][j] ** 2) / (2 * z)
                similarity_matrix[i][j] = math.exp(-coeff)
                similarity_matrix[j][i] = similarity_matrix[i][j]

    similarity_matrix_data = pd.DataFrame(similarity_matrix)

    similarity_matrix_data.to_csv("calculated_data/similarity_matrix.csv")

    return similarity_matrix


def calculate_diagonal_matrix(W):
    D = np.diag(W.sum(axis=1))

    D_data = pd.DataFrame(D)
    D_data.to_csv("calculated_data/diagonal_matrix.csv")

    return D


def calculate_regularized_laplacian_matrix(W, D):
    laplaican_matrix_L = D - W

    D_inv_sqrt = np.linalg.inv(D ** (1 / 2))

    regularized_L = np.dot(laplaican_matrix_L, D_inv_sqrt)
    regularized_L = np.dot(D_inv_sqrt, regularized_L)

    regularized_L_data = pd.DataFrame(regularized_L)
    regularized_L_data.to_csv("calculated_data/regularized_laplacian_matrix.csv")

    return regularized_L


corr_matrix_R = calculate_corr_matrix()

WMD_matrix = calculate_WMD(corr_matrix_R)

similarity_matrix_W = calculate_similarity_matrix(WMD_matrix, 3)

diagonal_matrix_D = calculate_diagonal_matrix(similarity_matrix_W)

regularized_laplacian_matrix_L = calculate_regularized_laplacian_matrix(
    similarity_matrix_W, diagonal_matrix_D
)

print("Done")

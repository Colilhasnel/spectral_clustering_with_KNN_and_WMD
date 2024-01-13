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


def calculate_cov_matrix_corr_matrix():
    cov_matrix = np.zeros((P, P), dtype=np.float64)

    corr_matrix = np.zeros((P, P), dtype=np.float64)

    for i in range(0, P):
        pi = VARIABLES[i]
        for j in range(i, P):
            pj = VARIABLES[j]
            for k in range(0, N):
                cov_matrix[i][j] += (DATA[pi][k] - MEAN[pi]) * (DATA[pj][k] - MEAN[pi])
            cov_matrix[i][j] = cov_matrix[i][j] / (N - 1)
            corr_matrix[i][j] = cov_matrix[i][j] / (
                STANDARD_DEVIATION[pi] * STANDARD_DEVIATION[pj]
            )

    for i in range(0, P):
        for j in range(0, i):
            cov_matrix[i][j] = cov_matrix[j][i]
            corr_matrix[i][j] = corr_matrix[j][i]

    return cov_matrix, corr_matrix


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
    knn_data.to_csv("k_nn_datafile.csv")

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

    similarity_matrix_data.to_csv("similarity_matrix_data.csv")


matrices = calculate_cov_matrix_corr_matrix()

print("Calculated T and R")

cov_matrix_T = matrices[0]
corr_matrix_R = matrices[1]

WMD_matrix = calculate_WMD(corr_matrix_R)

print("Calculated WMD")

calculate_similarity_matrix(WMD_matrix, 3)

print("Done")


# print(np.array(DATA.loc[[1]]))

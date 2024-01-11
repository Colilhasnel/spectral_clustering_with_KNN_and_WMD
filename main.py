import os
import numpy as np
import pandas as pd
import math

data = pd.read_csv("newdataset_spectral_clustering.dat", delimiter="\s+")

data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

data = data.dropna()

data.reset_index(inplace=True)

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

data = data[selected_columns]


def calculate_mahalanobis(data):
    N = len(data.axes[0])
    P = len(data.axes[1])

    mean = data.mean(axis=0)

    standard_deviation = data.std()

    variables = list(data.columns)

    cov_matrix = np.zeros((P, P), dtype=np.float64)

    corr_matrix = np.zeros((P, P), dtype=np.float64)

    for i in range(0, P):
        pi = variables[i]
        for j in range(0, P):
            pj = variables[j]
            if i >= j:
                for k in range(0, N):
                    cov_matrix[i][j] += (data[pi][k] - mean[pi]) * (
                        data[pj][k] - mean[pi]
                    )
                cov_matrix[i][j] = cov_matrix[i][j] / (N - 1)
                corr_matrix[i][j] = cov_matrix[i][j] / (
                    standard_deviation[pi] * standard_deviation[pj]
                )
            else:
                continue

    for i in range(0, P):
        for j in range(0, P):
            if i < j:
                cov_matrix[i][j] = cov_matrix[j][i]
                corr_matrix[i][j] = corr_matrix[j][i]

    WMD = np.zeros((N, N), dtype=np.float64)

    X = np.array(data.loc[3])
    Y = np.array(data.loc[5])

    X = np.transpose(X)
    Y = np.transpose(Y)

    result = np.dot(corr_matrix, X - Y)

    result = np.dot(np.transpose(X - Y), result)

    print(result)

    for i in range(0, N):
        for j in range(0, N):
            if i >= j:
                X = np.array(data.loc[i])
                Y = np.array(data.loc[j])

                X = np.transpose(X)
                Y = np.transpose(Y)

                ans = np.dot(corr_matrix, X - Y)
                ans = np.dot(np.transpose(X - Y), ans)
                ans = math.sqrt(ans)

                WMD[i][j] = ans

            else:
                continue

    for i in range(0, N):
        for j in range(0, N):
            if i < j:
                WMD[i][j] = WMD[j][i]

    return WMD


# print(np.array(data.loc[[1]]))
calculate_mahalanobis(data)

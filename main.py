import os
import numpy as np
import pandas as pd

data = pd.read_csv("newdataset_spectral_clustering.dat", delimiter="\s+")

data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

data = data.dropna()

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

    indexes = list(data.index)

    mean = data.mean(axis=0)

    standard_deviation = data.std()

    variables = list(data.columns)

    cov_matrix = np.array([[0] * P] * P)

    corr_matrix = np.array([[0] * P] * P)

    # WMD = np.array()

    for pi in variables:
        i = variables.index(pi)
        for pj in variables:
            j = variables.index(pj)
            for k in indexes:
                cov_matrix[i][j] += (data[pi][k] - mean[pi]) * (data[pj][k] - mean[pi])
            cov_matrix[i][j] = cov_matrix[i][j] / (N - 1)
            corr_matrix[i][j] = cov_matrix[i][j] / (
                standard_deviation[pi] * standard_deviation[pj]
            )


# print(data.head())

calculate_mahalanobis(data)

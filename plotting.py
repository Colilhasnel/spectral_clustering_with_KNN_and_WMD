import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = pd.read_csv("newdataset_spectral_clustering.dat", delimiter="\s+")

DATA.columns = DATA.columns.str.strip()
DATA.replace([np.inf, -np.inf], np.nan, inplace=True)
DATA.dropna(inplace=True)

DATA = DATA.dropna()

DATA.reset_index(inplace=True)

VARIABLES = [
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

DATA = DATA[VARIABLES]


def plot_clusters(variable_1, variable_2, labels_file, k):
    labels = pd.read_csv(labels_file, index_col=0)

    colors = ["red", "blue", "green", "yellow", "black"]

    for i in range(0, k):
        x = DATA.loc[labels["Labels"] == i, variable_1].to_list()
        y = DATA.loc[labels["Labels"] == i, variable_2].to_list()

        plt.scatter(x, y, c=colors[i])

    plt.xlabel(variable_1)
    plt.ylabel(variable_2)

    plt.show()


plot_clusters(VARIABLES[0], VARIABLES[-4], "calculated_data/cluster_prediction.csv", 3)

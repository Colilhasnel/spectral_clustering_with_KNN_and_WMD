from sklearn import manifold
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
import get_calculated_data as data_storred
import pandas as pd
import numpy as np

n_components = 2

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
data = np.array(DATA)

# WMD = data_storred.get_data("WMD_matrix.csv")


# def add_2d_scatter(ax, points, points_color, title=None):
#     x, y = points.T
#     ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
#     ax.set_title(title)
#     ax.xaxis.set_major_formatter(ticker.NullFormatter())
#     ax.yaxis.set_major_formatter(ticker.NullFormatter())


# def plot_2d(points, points_color, title):
#     fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
#     fig.suptitle(title, size=16)
#     add_2d_scatter(ax, points, points_color)
#     plt.show()


def plot_clusters(tsne_data, labels_file, k):
    labels = pd.read_csv(labels_file, index_col=0)

    # print(labels.iloc[2, 0])

    colors = ["red", "blue", "green", "yellow", "black"]

    for i in range(0, k):
        x = tsne_data[i][0]
        y = tsne_data[i][1]
        # x = DATA.loc[labels["Labels"] == i, variable_1].to_list()
        # y = DATA.loc[labels["Labels"] == i, variable_2].to_list()

        plt.scatter(x, y, c=colors[labels.iloc[i, 0]])

    plt.show()


t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=30,
    init="random",
    n_iter=250,
    random_state=0,
)
S_t_sne = t_sne.fit_transform(data)

plot_clusters(S_t_sne, "calculated_data/cluster_prediction.csv", S_t_sne.shape[0])

# plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")

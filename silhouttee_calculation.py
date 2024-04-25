import os
import numpy as np
import pandas as pd
import math
import get_calculated_data as data_storred
from sklearn.metrics import silhouette_score

DATA = pd.read_csv("newDATAset_spectral_clustering.dat", delimiter="\s+")

DATA.columns = DATA.columns.str.strip()
DATA.replace([np.inf, -np.inf], np.nan, inplace=True)
DATA.dropna(inplace=True)

DATA = DATA.dropna()

DATA.reset_index(inplace=True)

WMD = data_storred.get_data("WMD_matrix.csv")

labels_asigned = data_storred.get_data("cluster_prediction.csv")
labels_asigned = labels_asigned.reshape(1, -1)[0]

# print(labels_asigned)

value = silhouette_score(X=WMD, labels=labels_asigned, metric="precomputed")

print(value)

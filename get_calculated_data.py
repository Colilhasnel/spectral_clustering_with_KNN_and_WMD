import os
import pandas as pd
import numpy as np


def check_data(data_file):
    file_path = os.path.join("calculated_data", data_file)
    if os.path.exists(file_path):
        return True
    else:
        return False


def get_data(data_file):
    file_path = os.path.join("calculated_data", data_file)

    file = pd.read_csv(file_path, index_col=0)

    data = np.array(file, dtype=np.float64)

    return data

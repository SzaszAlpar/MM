import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def read_census():
    df = pd.read_csv('../FixedGroupSizeMM/adult.data', header=None)
    # df = df[[0, 2, 4, 10, 11, 12]]
    # column_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    df = df[[0,  10,  12]]
    column_names = ['age',  'capital-gain', 'hours-per-week']
    df.columns = column_names
    scaler = StandardScaler()
    for columnName in column_names:
        scaled_data = scaler.fit_transform(df[columnName].to_numpy().reshape(-1, 1))
        df[columnName] = scaled_data

    df = df.fillna(0).to_numpy()
    return df


def read_census_full():
    df = pd.read_csv('../FixedGroupSizeMM/adult.data', header=None)
    df = df[[0, 2, 4]]
    column_names = ['age', 'fnlwgt', 'education-num']
    df.columns = column_names
    # df = df.fillna(0).to_numpy()
    return df


def save_distance_matrix_to_disc():
    data = read_census()
    n = data.shape[0]
    dist_matrix = np.memmap('distance_matrix.dat', dtype='float32', mode='w+', shape=(n, n))

    # Calculate the distance matrix in chunks
    chunk_size = 1000  # Choose an appropriate chunk size based on your available memory
    for i in range(0, n, chunk_size):
        print(i, ". loop")
        end_i = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            dist_matrix[i:end_i, j:end_j] = distance_matrix(data[i:end_i], data[j:end_j])

    print("done creating distance matrix, now saving it")

    # Ensure all changes are written to disk
    dist_matrix.flush()


def plot_some_results():
    df = read_census_full()
    n = len(df)
    labels = np.memmap('../VariableGroupSizeMM/large_array.dat', dtype='float32', mode='r', shape=(n,))
    df['Cluster'] = labels
    custom_palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    sns.pairplot(df, hue='Cluster',palette=custom_palette)
    plt.show()


def main():
    records = read_census()
    print(records)


if __name__ == "__main__":
    plot_some_results()

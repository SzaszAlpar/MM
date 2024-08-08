import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

class Interpreter:
    def __init__(self, groups, centroids, scalers):
        self.groups = groups
        self.centroids = centroids
        self.column_names = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
                             'Heart Rate', 'Daily Steps']
        self.df_centroids = self.calculate_df_centroids(scalers)
        self.full_groups = None

    def calculate_df_centroids(self, scalers):
        aggregated_array = np.vstack(self.centroids)
        df2 = pd.DataFrame(aggregated_array, columns=self.column_names)
        for column in self.column_names:
            df2[column] = scalers[column].inverse_transform(df2[column].to_numpy().reshape(-1, 1))

        return df2

    def plot_centroids_PCA(self):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.df_centroids[self.column_names])

        df3 = pd.DataFrame()
        df3['PCA1'] = pca_result[:, 0]
        df3['PCA2'] = pca_result[:, 1]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PCA1', y='PCA2', data=df3)
        plt.title('PCA of Aggregated Data')
        plt.show()

    def plot_two_column_of_centroids(self, column1, column2):
        df3 = pd.DataFrame()
        df3[column1] = self.df_centroids[column1]
        df3[column2] = self.df_centroids[column2]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=column1, y=column2, data=df3)
        plt.title('Centroids based on the ' + column1 + ' and ' + column2)
        plt.show()

    def set_full_groups(self, full_data, result):
        groups = []
        groups_numbers = np.unique(result)
        for group in groups_numbers:
            group_data = full_data[result == group]
            groups.append(group_data)
        self.full_groups = groups

    # print the count of values in given column / groups
    def print_group_analysis(self, column_numbers):
        for i, group in enumerate(self.full_groups):
            print(i, ". group :")
            for column_number in column_numbers:
                values = np.array([v[column_number] for v in group])
                unique, counts = np.unique(values, return_counts=True)
                print(dict(zip(unique, counts)))

    def calculate_homogeneity(self):
        print("there are ", len(self.groups), "groups")
        total_hom = 0
        for group in self.groups:
            centroid = np.mean(group, axis=0)
            total_hom += np.sum((group - centroid) ** 2)
        print("total homogeneity is:", total_hom)

def get_centroid(records, nn):
    record_number = len(records)
    centroid = np.zeros(nn)
    if record_number == 0:
        return centroid
    else:
        return np.average(records, axis=0)


def aggregate(groups):
    result = []
    nn = len(groups[0][0])
    for group in groups:
        result.append(get_centroid(group, nn))
    return result


def aggregate(data, best_result):
    groups = []
    groups_numbers = np.unique(best_result)
    for group in groups_numbers:
        group_data = data[best_result == group]
        groups.append(group_data)
    result = []
    nn = len(groups[0][0])
    for group in groups:
        result.append(get_centroid(group, nn))
    return result


def get_groups(data, best_result):
    groups = []
    groups_numbers = np.unique(best_result)
    for group in groups_numbers:
        group_data = data[best_result == group]
        groups.append(group_data)
    return groups

def get_full_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df2 = df[
        ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Daily Steps']]

    return df2

def plot_some_results(labels):
    df = get_full_data()
    n = len(df)
    df['Cluster'] = labels
    custom_palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    # sns.pairplot(df, hue='Cluster',palette=custom_palette)
    sns.pairplot(df,kind='reg', hue='Cluster',palette=custom_palette,diag_kind='kde')
    plt.show()
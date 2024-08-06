import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from FixedGroupSizeMM import ResultInterpreter


def read_data_normalized():
    df = pd.read_csv('../FixedGroupSizeMM/Sleep_health_and_lifestyle_dataset.csv')
    df2 = df[
        ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]

    column_names = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate',
                    'Daily Steps']
    scalers = {}
    for column in column_names:
        scaler = StandardScaler()
        df2[column] = scaler.fit_transform(df2[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler

    df2 = df2.fillna(0).to_numpy()
    return [df2, scalers, df.fillna(0).to_numpy()]


# we try to minimalize the inf loss
def information_loss(clusters):
    loss = 0
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        loss += np.sum((cluster - centroid) ** 2)
    return loss


def combine(list1, list2):
    list1 = list(list1)
    list2 = list(list2)
    return np.array(list1 + list2)


def merge_clusters(cluster_list, small_cluster):
    centroid = small_cluster.mean(axis=0).tolist()
    centroids = [cluster.mean(axis=0).tolist() for cluster in cluster_list]
    # Merge clusters if they are smaller than k to the closest cluster
    distances = np.linalg.norm(np.array(centroids) - np.array(centroid), axis=1)
    closest_index = np.argmin(distances)

    cluster_list[closest_index] = combine(cluster_list[closest_index], small_cluster)

    return cluster_list, closest_index


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


def kmeans_microaggregation(data, k):
    n = data.shape[0]
    K = n // k  # Maximum number of clusters
    clusters = [data]
    global_clusters_idx = np.zeros(n, dtype=int)
    number_of_stagnation = 0
    cluster_length = 0

    while len(clusters) < K:
        print("############### Another iteration!\n")
        print("clusters length: ", len(clusters))
        for i, cluster in enumerate(clusters):
            print(i, ". groups size", len(cluster))
        best_loss = float('inf')
        best_idx1 = None
        best_idx2 = None
        best_cl1 = None
        best_cl2 = None
        best_clusters = None
        last_cluster = len(clusters)

        for i, cluster in enumerate(clusters):
            clusters_idx = np.where(global_clusters_idx == i)[0]

            # If the cluster is already close to size k it should not be divided
            if len(cluster) < 2 * k:
                continue

            kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(cluster)
            labels = kmeans.labels_

            new_clusters = [cluster[labels == 0], cluster[labels == 1]]
            idx1 = clusters_idx[labels == 0]
            idx2 = clusters_idx[labels == 1]

            if len(new_clusters[0]) >= k and len(new_clusters[1]) >= k:
                result_clusters = clusters[:i] + [new_clusters[0]] + clusters[i + 1:] + [new_clusters[1]]
                cl1 = i
                cl2 = last_cluster

            elif len(new_clusters[0]) < k and len(new_clusters[1]) < k:
                continue
            elif len(new_clusters[0]) >= k > len(new_clusters[1]):
                print("CASE1")
                result_clusters, idx = merge_clusters(clusters[:i] + clusters[i + 1:], new_clusters[1])
                result_clusters = result_clusters[:i] + [new_clusters[0]] + result_clusters[i:]
                # result_clusters.append(new_clusters[0])
                cl1 = i
                cl2 = idx + 1 if idx >= i else idx

            else:
                print("CASE2")
                result_clusters, idx = merge_clusters(clusters[:i] + clusters[i + 1:], new_clusters[0])
                result_clusters = result_clusters[:i] + [new_clusters[1]] + result_clusters[i:]
                # result_clusters.append(new_clusters[1])
                cl1 = idx + 1 if idx >= i else idx
                cl2 = i

            loss = information_loss(result_clusters)
            if loss < best_loss:
                best_loss = loss
                best_clusters = result_clusters
                best_idx1 = idx1
                best_idx2 = idx2
                best_cl1 = cl1
                best_cl2 = cl2

        if best_clusters is None:
            break

        clusters = best_clusters
        global_clusters_idx[best_idx1] = best_cl1
        global_clusters_idx[best_idx2] = best_cl2

        # in case that there are 2 clusters that are passing the same records to each other
        if len(clusters) == cluster_length:
            number_of_stagnation += 1
            if number_of_stagnation == 5:
                break
        else:
            cluster_length = len(clusters)

    return clusters, global_clusters_idx


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    [records, sc, full_data] = read_data_normalized()
    k = 25
    n_clusters = len(records) // k
    print("Maximum number of clusters: ", n_clusters)

    groups, result_idx = kmeans_microaggregation(records, k)
    print("Actual number of clusters: ", len(groups))

    for i, gr in enumerate(groups):
        print(i, ". group size: ", len(gr))

    centroids = aggregate(groups)
    print("centroids:", centroids)
    RI = ResultInterpreter.Interpreter(groups, centroids, sc)
    RI.set_full_groups(full_data, result_idx)
    RI.print_group_analysis([3, 8, 2])
    RI.plot_two_column_of_centroids('Quality of Sleep', 'Stress Level')
    RI.plot_two_column_of_centroids('Physical Activity Level', 'Daily Steps')
    RI.calculate_homogeneity()


if __name__ == "__main__":
    main()

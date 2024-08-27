import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from FixedGroupSizeMM import calculate_inf_loss


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


def keep_k_closest_to_centroid(group, k, centroid):
    distances = np.linalg.norm(group - centroid, axis=1)
    sorted_indices = np.argsort(distances)
    return group[sorted_indices[:k]], sorted_indices[:k], group[sorted_indices[k:]], sorted_indices[k:]


def keep_k_from_the_smallest_cluster(smaller_cl, bigger_cl, k, centroid, idx1, idx2):
    new_cls1, new_cls1_ix, remaining, remaining_ix = keep_k_closest_to_centroid(smaller_cl, k, centroid)
    new_cls2 = np.vstack([bigger_cl, remaining])
    new_idx2 = np.append(idx2, idx1[remaining_ix])
    new_idx1 = idx1[new_cls1_ix]

    return new_cls1, new_idx1, new_cls2, new_idx2


def kmeans_microaggregation(data, k):
    n = data.shape[0]
    K = n // k  # Maximum number of clusters
    clusters = [data]
    global_clusters_idx = np.zeros(n, dtype=int)
    number_of_stagnation = 0
    cluster_length = 0

    while len(clusters) < K:
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
            centroids = kmeans.cluster_centers_

            new_clusters = [cluster[labels == 0], cluster[labels == 1]]
            idx1 = clusters_idx[labels == 0]
            idx2 = clusters_idx[labels == 1]

            if len(new_clusters[0]) >= k and len(new_clusters[1]) >= k:
                if len(new_clusters[0]) <= len(new_clusters[1]):
                    new_clusters[0], idx1, new_clusters[1], idx2 = (
                        keep_k_from_the_smallest_cluster(new_clusters[0], new_clusters[1], k, centroids[0], idx1, idx2))
                else:
                    new_clusters[0], idx1, new_clusters[1], idx2 = (
                        keep_k_from_the_smallest_cluster(new_clusters[1], new_clusters[0], k, centroids[1], idx2, idx1))

                result_clusters = clusters[:i] + [new_clusters[0]] + clusters[i + 1:] + [new_clusters[1]]
                cl1 = i
                cl2 = last_cluster

            elif len(new_clusters[0]) < k and len(new_clusters[1]) < k:
                continue
            elif len(new_clusters[0]) >= k > len(new_clusters[1]):
                # print("CASE1")
                new_clusters[0], idx1, new_clusters[1], idx2 = (
                    keep_k_from_the_smallest_cluster(new_clusters[0], new_clusters[1], k, centroids[0], idx1, idx2))

                if len(new_clusters[1]) >= k:
                    result_clusters = clusters[:i] + [new_clusters[0]] + clusters[i + 1:] + [new_clusters[1]]
                    cl1 = i
                    cl2 = last_cluster
                else:
                    result_clusters, idx = merge_clusters(clusters[:i] + clusters[i + 1:], new_clusters[1])
                    result_clusters = result_clusters[:i] + [new_clusters[0]] + result_clusters[i:]
                    cl1 = i
                    cl2 = idx + 1 if idx >= i else idx

            else:
                # print("CASE2")
                new_clusters[1], idx2, new_clusters[0], idx1 = (
                    keep_k_from_the_smallest_cluster(new_clusters[1], new_clusters[0], k, centroids[1], idx2, idx1))
                if len(new_clusters[0]) >= k:
                    result_clusters = clusters[:i] + [new_clusters[1]] + clusters[i + 1:] + [new_clusters[0]]
                    cl1 = last_cluster
                    cl2 = i
                else:
                    result_clusters, idx = merge_clusters(clusters[:i] + clusters[i + 1:], new_clusters[0])
                    result_clusters = result_clusters[:i] + [new_clusters[1]] + result_clusters[i:]
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


def main2():
    pd.options.mode.chained_assignment = None

    dt_barcelona = '../Datasets/barcelona.csv'
    dt_Census = '../Datasets/Census.csv'
    dt_EIA = '../Datasets/EIA.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_tarraco = '../Datasets/tarraco.csv'
    dt_tarragona = '../Datasets/tarragona.csv'
    datasets1 = [dt_madrid, dt_tarraco, dt_barcelona]
    datasets2 = [dt_tarragona, dt_Census, dt_EIA]

    for p in range(3):
        print(p, ". ITERATION")
        for df in datasets1:
            for k in range(3, 6):
                print("working on dataset" + df)
                print("k=", k)
                records = calculate_inf_loss.read_dataset_wo_header(df)
                groups, result_idx = kmeans_microaggregation(records, k)
                calculate_inf_loss.calculate_I_loss(records, result_idx)

        for df in datasets2:
            for k in range(3, 6):
                print("working on dataset" + df)
                print("k=", k)
                records = calculate_inf_loss.read_dataset(df)
                groups, result_idx = kmeans_microaggregation(records, k)
                calculate_inf_loss.calculate_I_loss(records, result_idx)


if __name__ == "__main__":
    main2()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from FixedGroupSizeMM import calculate_inf_loss


def read_dataset_wo_header(path):
    df = pd.read_csv(path, sep=';', header=None)

    scalers = {}
    for column in range(len(df.columns)):
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler
    df = df.fillna(0).to_numpy()
    return df


def read_dataset(path):
    df = pd.read_csv(path)
    columns = df.columns
    scalers = {}

    for column in columns:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler
    df = df.fillna(0).to_numpy()
    return df


def normalize_data(data, overall_min, overall_max):
    return 2 * (data - overall_min) / (overall_max - overall_min) - 1


def fitness_function(centroids, data, k):
    assignments = assign_data_to_clusters(data, centroids, k)

    quantization_error = 0
    for j in range(len(centroids)):
        assigned_points = data[assignments == j]
        squared_distances = np.linalg.norm(assigned_points - centroids[j], axis=1) ** 2
        quantization_error += np.sum(squared_distances)
    return quantization_error


def reassign_points(assignments, centroids, k):
    while True:
        # if any cluster is smaller than k, we reassign the clusters
        cluster_sizes = np.bincount(assignments, minlength=len(centroids))
        if np.all(cluster_sizes >= k):
            break

        small_clusters = np.where(cluster_sizes < k)[0]
        for cluster in small_clusters:
            largest_cluster = np.argmax(cluster_sizes)
            points_in_largest_cluster = np.where(assignments == largest_cluster)[0]
            points_to_reassign = points_in_largest_cluster[:k - cluster_sizes[cluster]]
            assignments[points_to_reassign] = cluster
            cluster_sizes = np.bincount(assignments, minlength=len(centroids))
    return assignments


def assign_data_to_clusters(data, centroids, k):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    assignments = np.argmin(distances, axis=1)
    cluster_sizes = np.bincount(assignments, minlength=len(centroids))

    for cluster_index in range(len(centroids)):
        if cluster_sizes[cluster_index] < k:
            needed_points = k - cluster_sizes[cluster_index]

            centroid_distances = np.linalg.norm(centroids - centroids[cluster_index], axis=1)
            sorted_cluster_indices = np.argsort(centroid_distances)

            for other_cluster in sorted_cluster_indices:
                if cluster_index == other_cluster:
                    continue

                excess_points = cluster_sizes[other_cluster] - k
                if excess_points > 0:
                    # A szomszed klaszter legkulsobb tagjait valasztjuk le
                    candidates = np.where(assignments == other_cluster)[0]
                    sorted_candidates = np.argsort(distances[candidates, other_cluster])[::-1]
                    move_indices = candidates[sorted_candidates[:excess_points]]

                    assignments[move_indices] = cluster_index
                    cluster_sizes[cluster_index] += len(move_indices)
                    cluster_sizes[other_cluster] -= len(move_indices)
                    needed_points -= len(move_indices)
                    if needed_points <= 0:
                        break
    return assignments


def initialize_particles(data, num_particles, num_clusters, num_dimensions):
    num_kmeans_particles = num_particles // 2

    kmeans = KMeans(n_clusters=num_clusters).fit(data)
    centroids = kmeans.cluster_centers_
    particles = np.zeros((num_particles, num_clusters, num_dimensions))
    for i in range(num_kmeans_particles):
        jitter = np.random.normal(0, 0.1, centroids.shape)
        particles[i] = np.clip(centroids + jitter, 0, 1)

    for i in range(num_kmeans_particles, num_particles):
        particles[i] = np.random.rand(num_clusters, num_dimensions)

    return particles


def PSO_standard(data, k, num_particles, max_iterations, c1, c2, num_dimensions, num_clusters, particles):
    w_max = 0.9
    w_min = 0.4
    velocities = np.random.rand(num_particles, num_clusters, num_dimensions) * 0.1

    pBests = particles.copy()
    pBest_values = np.array([fitness_function(p, data, k) for p in pBests])
    gBest = pBests[np.argmin(pBest_values)]
    gBest_value = np.min(pBest_values)
    stagnation_threshold = 0.01
    improvement_threshold = 0.001
    w = 0.72

    for iteration in range(max_iterations):
        # w = w_max - ((w_max - w_min) * iteration / max_iterations)
        c1_dynamic = c1 + iteration / max_iterations * 0.5  # Increase c1 slightly over time
        c2_dynamic = c2 - iteration / max_iterations * 0.5  # Decrease c2 slightly over time
        # c1_dynamic = c1 - (iteration / max_iterations) * (c1 - 1.0)
        # c2_dynamic = c2 + (iteration / max_iterations) * (2.0 - c2)
        improvement = np.abs(gBest_value - np.min(pBest_values))

        # Check for stagnation and adjust velocities
        if improvement < improvement_threshold:
            velocities += np.random.rand(num_particles, num_clusters, num_dimensions) * stagnation_threshold

        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1_dynamic * r1 * (pBests[i] - particles[i]) +
                             c2_dynamic * r2 * (gBest - particles[i]))
            particles[i] += velocities[i]

            # Ensure particles stay within bounds
            particles[i] = np.clip(particles[i], 0, 1)

            current_fitness = fitness_function(particles[i], data, k)
            if current_fitness < pBest_values[i]:
                pBests[i] = particles[i]
                pBest_values[i] = current_fitness
                if current_fitness < gBest_value:
                    gBest = particles[i]
                    gBest_value = current_fitness
        if iteration % 10 == 0:
            print("iteration ", iteration)
            # print("current best", gBest_value)
    return [gBest, gBest_value]


def PSO(dt_tuple, k, num_particles, max_iterations, c1, c2):
    name, category = dt_tuple
    if category:
        data = read_dataset(name)
    else:
        data = read_dataset_wo_header(name)
    num_dimensions = data.shape[1]
    num_clusters = len(data) // k
    particles = np.random.rand(num_particles, num_clusters, num_dimensions)
    return PSO_standard(data, k, num_particles, max_iterations, c1, c2, num_dimensions, num_clusters, particles)


def PSO_initalized_with_kmean(data, k, num_particles, max_iterations, c1, c2):
    overall_min = np.min(data)
    overall_max = np.max(data)
    data = normalize_data(data, overall_min, overall_max)
    num_dimensions = data.shape[1]
    num_clusters = len(data) // k
    particles = initialize_particles(data, num_particles, num_clusters, num_dimensions)
    return PSO_standard(data, k, num_particles, max_iterations, c1, c2, num_dimensions, num_clusters, particles)


def main2():
    pd.options.mode.chained_assignment = None  # default='warn'
    dt_barcelona = '../Datasets/barcelona.csv'
    dt_Census = '../Datasets/Census.csv'
    dt_EIA = '../Datasets/EIA.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_tarraco = '../Datasets/tarraco.csv'
    dt_tarragona = '../Datasets/tarragona.csv'

    datasets1 = [dt_barcelona, dt_madrid, dt_tarraco]
    datasets2 = [dt_tarragona, dt_Census, dt_EIA]
    kx = [3, 4, 5]

    for df in datasets1:
        for k in kx:
            # k = 5
            print("working on" + df + " with k=", k)
            num_particles = 50
            max_iterations = 500
            c1 = 1.49  # Cognitive constant
            c2 = 1.49  # Social constant
            [gBest, gBest_value] = PSO([df, 0], k, num_particles, max_iterations, c1, c2)

            print("gbest value:", gBest_value)

            records = calculate_inf_loss.read_dataset_wo_header(df)
            group_assignments = assign_data_to_clusters(records, gBest, k)
            calculate_inf_loss.calculate_I_loss(records, group_assignments)

    for df in datasets2:
        for k in kx:
            # k = 5
            print("working on" + df + " with k=", k)
            num_particles = 10
            max_iterations = 400
            c1 = 1.49  # Cognitive constant
            c2 = 1.49  # Social constant
            [gBest, gBest_value] = PSO([df, 1], k, num_particles, max_iterations, c1, c2)

            print("gbest value:", gBest_value)

            records = calculate_inf_loss.read_dataset(df)
            group_assignments = assign_data_to_clusters(records, gBest, k)
            calculate_inf_loss.calculate_I_loss(records, group_assignments)


if __name__ == "__main__":
    main2()

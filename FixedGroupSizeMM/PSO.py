import numpy as np
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from FixedGroupSizeMM import calculate_inf_loss


def read_testdata():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df2 = df[
        ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]
    column_names = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate',
                    'Daily Steps']
    scalers = {}
    for column in column_names:
        scaler = MinMaxScaler()
        df2[column] = scaler.fit_transform(df2[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler

    df2 = df2.fillna(0).to_numpy()
    return df2


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

    SSE = 0
    unique_groups = np.unique(assignments)
    for group in unique_groups:
        group_indices = np.where(assignments == group)
        group_data = data[group_indices]
        group_mean = np.mean(group_data, axis=0)
        SSE += np.sum((group_data - group_mean) ** 2)
    return SSE


def assign_data_to_clusters(data, centroids, k):
    n = data.shape[0]
    assignments = -1 * np.ones(n, dtype=int)
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    assigned_indices = set()

    for cluster_index in range(len(centroids)):
        closest_indices = np.argsort(distances[:, cluster_index])
        j = 0
        for idx in closest_indices:
            if idx not in assigned_indices:
                assignments[idx] = cluster_index
                assigned_indices.add(idx)
                j += 1
            if j == k:
                break

    remaining_indices = np.where(assignments == -1)[0]
    assignments[remaining_indices] = len(centroids) - 1
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


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    dt_barcelona = '../Datasets/barcelona.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_tarraco = '../Datasets/tarraco.csv'
    dt_tarragona = '../Datasets/tarragona.csv'
    dataset_Census = '../Datasets/Census.csv'
    dt_EIA = '../Datasets/EIA.csv'
    datasets1 = [dt_barcelona, dt_madrid, dt_tarraco]
    datasets2 = [dt_tarragona, dataset_Census, dt_EIA]
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


if __name__ == "__main__":
    main()

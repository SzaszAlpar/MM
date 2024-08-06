import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ResultInterpreter


def read_data_normalized():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
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


def fitness_function(centroids, data, k):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    assignments = np.argmin(distances, axis=1)
    assignments = reassign_points(assignments, centroids, k)

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


def get_centroid(records, nn):
    record_number = len(records)
    centroid = np.zeros(nn)
    if record_number == 0:
        return centroid
    else:
        return np.average(records, axis=0)


def aggregate(data, best_solution):
    result = []
    nn = data.shape[1]
    groups_numbers = np.unique(best_solution)
    for i in groups_numbers:
        group = data[best_solution == i]
        result.append(get_centroid(group, nn))
    return result


def PSO(data, k, num_particles, max_iterations, w, c1, c2):
    num_dimensions = data.shape[1]
    num_clusters = len(data) // k
    w_max = 0.9
    w_min = 0.4
    particles = np.random.rand(num_particles, num_clusters, num_dimensions)
    velocities = np.random.rand(num_particles, num_clusters, num_dimensions) * 0.1

    pBests = particles.copy()
    pBest_values = np.array([fitness_function(p, data, k) for p in pBests])
    gBest = pBests[np.argmin(pBest_values)]
    gBest_value = np.min(pBest_values)

    for iteration in range(max_iterations):
        print("iteration ", iteration)
        w = w_max - ((w_max - w_min) * iteration / max_iterations)
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pBests[i] - particles[i]) +
                             c2 * r2 * (gBest - particles[i]))
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
    return [gBest, gBest_value]


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    data = np.random.rand(20, 3)
    num_particles = 200
    k = 5
    max_iterations = 100
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive constant
    c2 = 1.5  # Social constant
    [gBest, gBest_value] = PSO(data, k, num_particles, max_iterations, w, c1, c2)

    print("gbest value:", gBest_value)
    group_assignments = np.argmin(np.linalg.norm(data[:, None] - gBest, axis=2), axis=1)
    print("group_assignments:", group_assignments)

    colors = group_assignments
    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='viridis')
    plt.scatter(gBest[:, 0], gBest[:, 1], c='red', marker='x', label='Centroids')
    plt.title('Data Points Grouped by PSO-based k-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


def main2():
    pd.options.mode.chained_assignment = None  # default='warn'
    [records, sc, full_data] = read_data_normalized()
    num_particles = 200
    k = 40
    max_iterations = 10
    w = 0.5  # Inertia weight
    c1 = 1.5  # Cognitive constant
    c2 = 1.5  # Social constant
    [gBest, gBest_value] = PSO(records, k, num_particles, max_iterations, w, c1, c2)

    print("gbest value:", gBest_value)

    distances = np.linalg.norm(records[:, None] - gBest, axis=2)
    group_assignments = np.argmin(distances, axis=1)
    group_assignments = reassign_points(group_assignments, gBest, k)
    print("group_assignments:", group_assignments)

    interpret_result(group_assignments, full_data, records, sc)


def interpret_result(best_solution, full_data, records, sc):
    centroids = ResultInterpreter.aggregate(records, best_solution)
    groups = ResultInterpreter.get_groups(records, best_solution)
    RI = ResultInterpreter.Interpreter(groups, centroids, sc)
    RI.set_full_groups(full_data, best_solution)
    RI.print_group_analysis([3, 8, 2])
    RI.plot_two_column_of_centroids('Quality of Sleep', 'Stress Level')
    RI.plot_two_column_of_centroids('Physical Activity Level', 'Daily Steps')
    RI.calculate_homogeneity()


if __name__ == "__main__":
    main2()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
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


def initialize_solution(n, k):
    cluster_numbers = n // k
    solution = np.zeros(n, dtype=int)
    cluster_sizes = [k] * (cluster_numbers)
    for i in range(n % k):
        cluster_sizes[random.choice(range(n // k))] += 1

    current_idx = 0
    for cluster_id in range(1, cluster_numbers + 1):
        for _ in range(cluster_sizes[cluster_id - 1]):
            solution[current_idx] = cluster_id
            current_idx += 1

    np.random.shuffle(solution)
    return solution


def generate_neighbor(solution):
    neighbor = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def calculate_energy(solution, data, num_clusters):
    energy = 0
    for cluster_id in range(1, num_clusters + 1):
        cluster_points = data[np.where(solution == cluster_id)]
        if len(cluster_points) > 0:
            mean = np.mean(cluster_points, axis=0)
            energy += np.sum((cluster_points - mean) ** 2)
    return energy


def simulated_annealing(data, num_clusters, initial_temperature, cooling_rate, max_iterations, min_energy_threshold):
    n = len(data)
    current_solution = initialize_solution(n, num_clusters)
    print("initial solution: ", current_solution)
    current_energy = calculate_energy(current_solution, data, num_clusters)
    print("initial energy ", current_energy)
    best_solution = current_solution.copy()
    best_energy = current_energy

    temperature = initial_temperature

    for iteration in range(max_iterations):
        if current_energy < min_energy_threshold:
            break
        neighbor_solution = generate_neighbor(current_solution)
        neighbor_energy = calculate_energy(neighbor_solution, data, num_clusters)

        if neighbor_energy < current_energy or random.random() < np.exp(
                (current_energy - neighbor_energy) / temperature):
            current_solution = neighbor_solution
            current_energy = neighbor_energy

        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy

        temperature *= cooling_rate

    return best_solution, best_energy


# if we use this method the clusters can have a smaller size than k
# def perturb_solution(solution, num_clusters):
#     # Perturb the solution by reassigning some points randomly to different clusters
#     perturbed_solution = solution.copy()
#     num_perturbations = len(solution) // 10  # Change 10% of the points
#     indices_to_perturb = random.sample(range(len(solution)), num_perturbations)
#     for idx in indices_to_perturb:
#         perturbed_solution[idx] = random.randint(1, num_clusters)
#     return perturbed_solution

def perturb_solution(solution):
    # Perturb the solution by reassigning some points randomly to different clusters
    perturbed_solution = solution.copy()
    num_perturbations = len(solution) // 10  # Change 10% of the points
    indices_to_perturb = random.sample(range(len(solution)), num_perturbations)
    indices_to_swap_with = random.sample(range(len(solution)), num_perturbations)
    for count, idx in enumerate(indices_to_perturb):
        aux = perturbed_solution[idx]
        perturbed_solution[idx] = perturbed_solution[indices_to_swap_with[count]]
        perturbed_solution[indices_to_swap_with[count]] = aux

    return perturbed_solution


def simulated_annealing2(data, k, initial_temperature, cooling_rate, max_iterations, min_energy_threshold,
                         max_stagnation_iterations):
    n = len(data)
    num_clusters = n // k
    current_solution = initialize_solution(n, k)
    current_energy = calculate_energy(current_solution, data, num_clusters)

    best_solution = current_solution.copy()
    best_energy = current_energy

    temperature = initial_temperature
    stagnation_counter = 0

    for iteration in range(max_iterations):
        if current_energy < min_energy_threshold:
            break

        neighbor_solution = generate_neighbor(current_solution)
        neighbor_energy = calculate_energy(neighbor_solution, data, num_clusters)

        if neighbor_energy < current_energy or random.random() < np.exp(
                (current_energy - neighbor_energy) / temperature):
            current_solution = neighbor_solution
            current_energy = neighbor_energy

        if current_energy < best_energy:
            best_solution = current_solution.copy()
            best_energy = current_energy
            stagnation_counter = 0  # Reset stagnation counter
        else:
            stagnation_counter += 1

        if stagnation_counter > max_stagnation_iterations:
            current_solution = perturb_solution(best_solution)
            current_energy = calculate_energy(current_solution, data, num_clusters)
            stagnation_counter = 0  # Reset stagnation counter after perturbation

        temperature *= cooling_rate

    return best_solution, best_energy


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    [records, sc, full_data] = read_data_normalized()
    k = 40
    initial_temperature = 1000
    cooling_rate = 0.99
    max_iterations = 2000
    min_energy_threshold = 1e-5  # Minimum energy threshold to avoid getting stuck
    max_stagnation_iterations = 100  # Maximum iterations without significant improvement

    best_solution, best_energy = simulated_annealing2(records, k, initial_temperature, cooling_rate,
                                                      max_iterations, min_energy_threshold, max_stagnation_iterations)

    print("Best Solution:", best_solution)
    print("Best Energy:", best_energy)
    interpret_result(best_solution, full_data, records, sc)



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
    main()

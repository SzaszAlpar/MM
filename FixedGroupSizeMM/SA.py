import random

import numpy as np
import pandas as pd

from FixedGroupSizeMM import calculate_inf_loss


def initialize_solution(n, k):
    cluster_numbers = n // k
    solution = np.zeros(n, dtype=int)
    cluster_sizes = [k] * cluster_numbers

    # The remaining records will be added to a random group
    cluster_sizes[random.choice(range(cluster_numbers))] += n % k

    current_idx = 0
    for cluster_id in range(cluster_numbers):
        for _ in range(cluster_sizes[cluster_id]):
            solution[current_idx] = cluster_id
            current_idx += 1

    np.random.shuffle(solution)
    return solution


def generate_neighbor(solution):
    neighbor = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def generate_neighbor2(solution):
    return perturb_solution(solution)


def generate_neighbor3(solution, probability_threshold=0.99):
    n = len(solution)
    flag = False
    i = 0

    while i < n or not flag:
        u = random.uniform(0, 1)

        if u > probability_threshold:
            flag = True
            i, j = random.sample(range(n), 2)
            solution[i], solution[j] = solution[j], solution[i]
        i += 1
        if i == n and flag:
            break
        if i == n:
            i = 0
    return solution


def calculate_energy(solution, data, num_clusters):
    SSE = 0
    unique_groups = np.unique(solution)
    for group in unique_groups:
        group_indices = np.where(solution == group)
        group_data = data[group_indices]
        group_mean = np.mean(group_data, axis=0)
        SSE += np.sum((group_data - group_mean) ** 2)
    return SSE


def perturb_solution(solution):
    # Perturb the solution by reassigning some points randomly to different clusters
    perturbed_solution = solution.copy()
    num_perturbations = len(solution) // 1  # Change 10% of the points
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
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter > max_stagnation_iterations:
            current_solution = perturb_solution(best_solution)
            current_energy = calculate_energy(current_solution, data, num_clusters)
            stagnation_counter = 0

        temperature *= cooling_rate
        # if iteration % 100 == 0:
        #     print("iteration", iteration)
        #     print("Current best energy:", best_energy)

    return best_solution, best_energy


def simulated_annealing22(data, k, initial_temperature, cooling_rate, max_iterations, min_energy_threshold,
                          max_stagnation_iterations, current_solution):
    n = len(data)
    num_clusters = n // k
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
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter > max_stagnation_iterations:
            current_solution = perturb_solution(best_solution)
            current_energy = calculate_energy(current_solution, data, num_clusters)
            stagnation_counter = 0

        temperature *= cooling_rate

    return best_solution, best_energy


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    dt_tarragona = '../Datasets/tarragona.csv'
    dt_EIA = '../Datasets/EIA.csv'
    dt_Census = '../Datasets/Census.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_barcelona = '../Datasets/barcelona.csv'
    dt_tarraco = '../Datasets/tarraco.csv'
    print("Working on dataset" + dt_EIA)
    ks = [3, 4, 5]
    for k in ks:
        for i in range(1):
            print("****************************")
            print(i, ". iteration!")
            records = calculate_inf_loss.read_dataset(dt_EIA)
            initial_temperature = 45
            cooling_rate = 0.90
            max_iterations = 20000
            min_energy_threshold = 1e-5  # Minimum energy threshold to avoid getting stuck
            max_stagnation_iterations = 400  # Maximum iterations without significant improvement

            best_solution, best_energy = simulated_annealing2(records, k, initial_temperature, cooling_rate,
                                                              max_iterations, min_energy_threshold,
                                                              max_stagnation_iterations)

            calculate_inf_loss.calculate_I_loss(records, best_solution)


if __name__ == "__main__":
    main()

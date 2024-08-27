import random

import numpy as np


def initialize_solution(n, k_min, k_max):
    solution = np.zeros(n, dtype=int)
    cluster_id = 1
    indices = np.arange(n)
    np.random.shuffle(indices)

    start_idx = 0
    while start_idx < n:
        remaining_points = n - start_idx
        current_k_max = min(k_max, remaining_points)
        current_k_min = min(k_min, remaining_points)

        if remaining_points <= k_max:
            cluster_size = remaining_points
        else:
            cluster_size = random.randint(current_k_min, current_k_max)

        end_idx = start_idx + cluster_size
        solution[indices[start_idx:end_idx]] = cluster_id
        cluster_id += 1
        start_idx = end_idx

    return solution


def generate_neighbor(solution):
    neighbor = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def calculate_energy(solution, data):
    unique_clusters = np.unique(solution)
    energy = 0
    for cluster_id in unique_clusters:
        cluster_points = data[np.where(solution == cluster_id)]
        if len(cluster_points) > 0:
            mean = np.mean(cluster_points, axis=0)
            energy += np.sum((cluster_points - mean) ** 2)
    return energy


def perturb_solution(solution):
    perturbed_solution = solution.copy()
    num_perturbations = len(solution) // 10
    indices_to_perturb = random.sample(range(len(solution)), num_perturbations)
    indices_to_swap_with = random.sample(range(len(solution)), num_perturbations)
    for count, idx in enumerate(indices_to_perturb):
        aux = perturbed_solution[idx]
        perturbed_solution[idx] = perturbed_solution[indices_to_swap_with[count]]
        perturbed_solution[indices_to_swap_with[count]] = aux

    return perturbed_solution


def simulated_annealing2(data, k_min, k_max, initial_temperature, cooling_rate, max_iterations, min_energy_threshold,
                         max_stagnation_iterations):
    n = len(data)
    current_solution = initialize_solution(n, k_min, k_max)
    current_energy = calculate_energy(current_solution, data)

    best_solution = current_solution.copy()
    best_energy = current_energy

    temperature = initial_temperature
    stagnation_counter = 0

    for iteration in range(max_iterations):
        if current_energy < min_energy_threshold:
            break

        neighbor_solution = generate_neighbor(current_solution)
        neighbor_energy = calculate_energy(neighbor_solution, data)

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
            current_energy = calculate_energy(current_solution, data)
            stagnation_counter = 0  # Reset stagnation counter after perturbation

        temperature *= cooling_rate

    return best_solution, best_energy

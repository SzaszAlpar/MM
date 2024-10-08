from FixedGroupSizeMM import MDAV
import numpy as np
from collections import defaultdict
from FixedGroupSizeMM import GA, SA, PSO, calculate_inf_loss
from VariableGroupSizeMM import PSO2, GA2, SA2


def get_SSE(result, data):
    SSE = 0
    unique_groups = np.unique(result)
    print("unique groups:", len(unique_groups))
    for group in unique_groups:
        group_indices = np.where(result == group)
        group_data = data[group_indices]
        group_mean = np.mean(group_data, axis=0)
        SSE += np.sum((group_data - group_mean) ** 2)
    return SSE


def MDAV_GA(records, k1, k2, population_zise, generations):
    n = len(records)
    final_assignments = -1 * np.ones(n, dtype=int)
    groups, indices = MDAV.MDAV(records, k1)

    idx = 0
    for i, group in enumerate(groups):
        print(i, ". th group from MDAV has ", len(group), " samples. Call GA on it, k=", k2)
        n_clusters = len(group) // k2

        best_solution, best_fitness = GA.genetic_algorithm(group, n_clusters, generations, k2, population_zise)
        indices_dict = defaultdict(list)
        for index, value in enumerate(best_solution):
            indices_dict[value].append(index)
        indices_grouped = [indices_dict[key] for key in sorted(indices_dict.keys())]

        small_group_sizes = [len(small_group) for small_group in indices_grouped]
        print("Graph resulted in ", len(indices_grouped), " smaller groups. Here are the sizes:", small_group_sizes)

        for small_group in indices_grouped:
            for small_group_idx in small_group:
                final_assignments[indices[i][small_group_idx]] = idx
            idx += 1
    print("final assignments", final_assignments)
    return final_assignments


def first_MDAAV_then_GA(records, k, population_size, generations):
    groups, indices = MDAV.MDAV(records, k)
    cluster_assignment = np.zeros(len(records), dtype=int)

    # Assign cluster labels
    for cluster_id, record_indices in enumerate(indices):
        for record_index in record_indices:
            cluster_assignment[record_index] = cluster_id

    print("MDAV is done, SSE value:", get_SSE(cluster_assignment, records))

    n_clusters = len(records) // k
    population = GA.initialize_population_with_given_value(population_size, len(records), n_clusters, k,
                                                           cluster_assignment)
    best_solution, best_fitness = GA.genetic_algorithm(records, n_clusters, generations, k, population_size, population)

    print("GA is done, SSE value:", get_SSE(best_solution, records))

    return best_solution


def first_SA_then_GA(records, k, population_size, generations):
    initial_temperature = 45
    cooling_rate = 0.90
    max_iterations = 30000
    min_energy_threshold = 1e-5  # Minimum energy threshold to avoid getting stuck
    max_stagnation_iterations = 400  # Maximum iterations without significant improvement

    best_solution, best_energy = SA.simulated_annealing2(records, k, initial_temperature, cooling_rate,
                                                         max_iterations, min_energy_threshold,
                                                         max_stagnation_iterations)

    print("SA is done, SSE value:", get_SSE(best_solution, records))

    n_clusters = len(records) // k
    population = GA.initialize_population_with_given_value(population_size, len(records), n_clusters, k,
                                                           best_solution)
    best_solution, best_fitness = GA.boosted_genetic_algorithm(records, n_clusters, generations, k, population_size, population)

    print("GA is done, SSE value:", get_SSE(best_solution, records))

    return best_solution


def first_PSO_then_GA(dt_tuple, k, population_size, generations):
    name, category = dt_tuple
    if category:
        records = calculate_inf_loss.read_dataset(name)
    else:
        records = calculate_inf_loss.read_dataset_wo_header(name)

    num_particles = 15
    max_iterations = 500
    c1 = 1.49  # Cognitive constant
    c2 = 1.49  # Social constant
    [gBest, gBest_value] = PSO.PSO(dt_tuple, k, num_particles, max_iterations, c1, c2)
    group_assignments = PSO.assign_data_to_clusters(records, gBest, k)

    print("PSA is done. Current results:")
    calculate_inf_loss.calculate_I_loss(records, group_assignments)

    n_clusters = len(records) // k
    population = GA.initialize_population_with_given_value(population_size, len(records), n_clusters, k,
                                                           group_assignments)
    best_solution, best_fitness = GA.boosted_genetic_algorithm(records, n_clusters, generations, k, population_size,
                                                               population)

    return best_solution


if __name__ == "__main__":
    dt_Census = '../Datasets/Census.csv'
    dt_EIA = '../Datasets/EIA.csv'
    dt_tarragona = '../Datasets/tarragona.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_barcelona = '../Datasets/barcelona.csv'
    dt_tarraco = '../Datasets/tarraco.csv'

    generations = 500
    population_size = 35

    datasets1 = [dt_barcelona, dt_madrid, dt_tarraco]
    datasets2 = [dt_tarragona, dt_Census, dt_EIA]
    kx = [3, 4, 5]


    # for k in kx:
    #     for dt in datasets1:
    #         print("*** WORKING ON:", dt)
    #         print("working on k=",k)
    #         records = calculate_inf_loss.read_dataset_wo_header(dt)
    #         fa = first_SA_then_GA(records, k, population_size, generations)
    #         calculate_inf_loss.calculate_I_loss(records, fa)

    for k in kx:
        for dt in datasets2:
            print("*** WORKING ON:", dt)
            print("working on k=",k)
            records = calculate_inf_loss.read_dataset(dt)
            fa = first_SA_then_GA(records, k, population_size, generations)
            calculate_inf_loss.calculate_I_loss(records, fa)

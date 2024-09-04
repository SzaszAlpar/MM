import random
import time

import numpy as np
import pandas as pd

from FixedGroupSizeMM import calculate_inf_loss


def initialize_population(size, chromosome_length, n_clusters, k, records):
    population = []
    for i in range(size):
        np.random.seed(int(time.time()) + i)
        chromosome = np.zeros(chromosome_length, dtype=int)

        # Assign each cluster number k times
        for cluster_num in range(n_clusters):
            indices = np.random.choice(np.where(chromosome == 0)[0], size=k, replace=False)
            chromosome[indices] = cluster_num

        rest = chromosome_length % k
        if rest != 0:
            remaining_indices = np.random.choice(np.where(chromosome == 0)[0], size=rest, replace=False)
            chosen_cluster = np.random.randint(0, n_clusters)
            chromosome[remaining_indices] = chosen_cluster

        population.append(chromosome)

    return population


def initialize_population_with_given_value(size, chromosome_length, n_clusters, k, given_value):
    nr = int(size * 0.2)
    population = []
    for j in range(nr):
        population.append(given_value)

    for i in range(size - nr):
        np.random.seed(int(time.time()) + i)
        chromosome = np.zeros(chromosome_length, dtype=int)

        for cluster_num in range(n_clusters):
            indices = np.random.choice(np.where(chromosome == 0)[0], size=k, replace=False)
            chromosome[indices] = cluster_num

        rest = chromosome_length % k
        if rest != 0:
            remaining_indices = np.random.choice(np.where(chromosome == 0)[0], size=rest, replace=False)
            chosen_cluster = np.random.randint(0, n_clusters)
            chromosome[remaining_indices] = chosen_cluster

        population.append(chromosome)

    return population


def get_parent_number(population, parent_rate=0.60):
    number = int(population * parent_rate)
    if number % 2 != 0:
        return number + 1
    else:
        return number


def shuffle_random_population(population, fitnesses, shuffle_percentage):
    num_to_shuffle = int(len(population) * shuffle_percentage)

    # Randomly select chromosomes to shuffle
    indices_to_shuffle = random.sample(range(len(population)), num_to_shuffle)

    for idx in indices_to_shuffle:
        np.random.shuffle(population[idx])

    return population


def fitness(chromosome, data, n_clusters):
    sse = 0
    unique_groups = np.unique(chromosome)
    for i in unique_groups:
        # we want to get all data points assigned to cluster i
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


# half of the chosen population is picked by elite mode
def elitism_selection(population, fitnesses, number_of_chromosomes):
    paired = list(zip(fitnesses, population))
    first_half = number_of_chromosomes // 5
    second_half = number_of_chromosomes - first_half
    indices = np.random.randint(first_half + 1, len(population), size=second_half)

    # Sort the population by fitness
    paired.sort(key=lambda x: x[0])
    selected_elite_mode = [ind for _, ind in paired[:first_half]]
    selected_random_mode = []
    for index in indices:
        _, chrr = paired[index]
        selected_random_mode.append(chrr)
    return selected_elite_mode + selected_random_mode


def tournament_selection(population, fitnesses, number_of_chromosomes, tournament_size=3):
    selected_pop = []
    available_indices = list(range(len(population)))

    for _ in range(number_of_chromosomes):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(available_indices, tournament_size)
        tournament_pop = [population[i] for i in tournament_indices]
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Choose the best individual from the tournament
        best_index = np.argmin(tournament_fitnesses)
        selected_pop.append(tournament_pop[best_index])
        available_indices.remove(tournament_indices[best_index])

    return selected_pop


def uniform_crossover(parent1, parent2, k, records, crossover_rate=0.50, parent_rate=0.5):
    if random.random() < crossover_rate:
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            random_nr = random.random()
            if random_nr < parent_rate:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        child1 = rearrange_chromosome_randomly(child1, k, records)
        child2 = rearrange_chromosome_randomly(child2, k, records)
        return np.array(child1), np.array(child2)
    return None, None


def rearrange_chromosome_randomly(chromosome, k, records):
    cluster_counts = {}
    for gene in chromosome:
        if gene in cluster_counts:
            cluster_counts[gene] += 1
        else:
            cluster_counts[gene] = 1

    small_clusters = [cluster for cluster, count in cluster_counts.items() if count < k]
    big_clusters = [cluster for cluster, count in cluster_counts.items() if count > k]

    while len(big_clusters) >= 1 and len(small_clusters) > 0:
        big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])

        while cluster_counts[big_cluster] > k and small_clusters:
            small_cluster = small_clusters[0]
            indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]
            random_index = random.choice(indices)
            chromosome[random_index] = small_cluster

            cluster_counts[small_cluster] += 1
            cluster_counts[big_cluster] -= 1

            if cluster_counts[big_cluster] == k:
                big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])

            if cluster_counts[small_cluster] == k:
                small_clusters.pop(0)
                continue
        big_clusters = [cluster for cluster in big_clusters if cluster_counts[cluster] > k]

    if len(big_clusters) > 1 and len(small_clusters) == 0:
        chromosome = rearrange_big_clusters_randomly(chromosome, k, cluster_counts, big_clusters, records)

    return chromosome


def rearrange_big_clusters_randomly(chromosome, k, cluster_counts, big_clusters, records):
    if len(big_clusters) > 1:
        chosen_cluster = random.choice(big_clusters)
        big_clusters.remove(chosen_cluster)

        for big_cluster in big_clusters:
            count = 0
            while cluster_counts[big_cluster] > k:
                indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]
                random_index = random.choice(indices)
                chromosome[random_index] = chosen_cluster
                cluster_counts[big_cluster] -= 1
                cluster_counts[chosen_cluster] += 1
                count += 1
                if (count > 100):
                    print("current cluster:", big_cluster)
                    print("cluster counts[big_cls]: ", cluster_counts[big_cluster])
                    print("indices: ", indices)
    return chromosome


def mutate(chromosome, records, curr_iteration, max_iteration):
    start = 0.5
    end = 0.01
    mutation_rate = start - ((start - end) / max_iteration) * curr_iteration
    # for i in range(len(chromosome)):
    if random.random() < mutation_rate:
        i = random.randint(0, len(records) - 1)
        swap_index = random.randint(0, chromosome.size - 1)
        while swap_index == i:
            swap_index = random.randint(0, chromosome.size - 1)

        aux = chromosome[i]
        chromosome[i] = chromosome[swap_index]
        chromosome[swap_index] = aux

    return chromosome


def genetic_algorithm(records, n_clusters, generations, k, population_size, population):
    max_stagnation = 50
    shuffle_percentage = 0.3
    fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
    best_solution = population[np.argmin(fitnesses)]
    best_fitness = min(fitnesses)
    stagnation_count = 0

    for generation in range(generations):

        fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= max_stagnation:
            fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
            population = shuffle_random_population(population, fitnesses, shuffle_percentage)
            stagnation_count = 0
        parent_number = get_parent_number(population_size)
        # select best parents to crossover
        parents = tournament_selection(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover(parent1, parent2, k, records)
            if child1 is not None and child2 is not None:
                offspring.append(mutate(child1, records, generation, generations))
                offspring.append(mutate(child2, records, generation, generations))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]

        population = elitism_selection(population, new_fitnesses, population_size)

        if generation % 100 == 0:
            print(generation, ". generation:")
            print("Generations fitnesses:", fitnesses)
            print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def uniform_crossover_distance_based(parent1, parent2, k, records, crossover_rate=0.90, parent_rate=0.5):
    if random.random() < crossover_rate:
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            random_nr = random.random()
            if random_nr < parent_rate:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        child1 = rearrange_chromosome_distance_based(child1, k, records)
        child2 = rearrange_chromosome_distance_based(child2, k, records)
        return np.array(child1), np.array(child2)
    return None, None


def rearrange_chromosome_distance_based(chromosome, k, records):
    n_clusters = len(records) // k

    cluster_counts = {i: 0 for i in range(n_clusters)}
    for gene in chromosome:
        cluster_counts[gene] += 1

    # print("len records", (cluster_counts))
    unique_groups = np.unique(chromosome)
    # print("unique count len", len(cluster_counts))

    small_clusters = [cluster for cluster, count in cluster_counts.items() if count < k]
    big_clusters = [cluster for cluster, count in cluster_counts.items() if count > k]

    while len(big_clusters) >= 1 and len(small_clusters) > 0:
        big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])
        # big_cluster = np.int64(big_cluster)

        while cluster_counts[big_cluster] > k and small_clusters:
            # small_cluster = small_clusters[0]
            # print("big_cluster:",big_cluster)
            chromosome_array = np.array(chromosome)
            # print("where",np.where(chromosome_array == big_cluster))
            # print("chromosome:", chromosome)

            # indices = np.where(chromosome == big_cluster)[0]
            indices = [index for index, elem in enumerate(chromosome) if elem == big_cluster]
            # print("indices:", indices)
            random_index = random.choice(indices)
            small_cluster = get_closest_records_cluster(chromosome, small_clusters, random_index, records)
            chromosome[random_index] = small_cluster

            # print("small_cluster",small_cluster)
            cluster_counts[small_cluster] += 1
            cluster_counts[big_cluster] -= 1

            if cluster_counts[big_cluster] == k:
                big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])

            if cluster_counts[small_cluster] == k:
                small_clusters.remove(small_cluster)
                continue
        big_clusters = [cluster for cluster in big_clusters if cluster_counts[cluster] > k]

    if len(big_clusters) > 1 and len(small_clusters) == 0:
        chromosome = rearrange_big_clusters_distance_based(chromosome, k, cluster_counts, big_clusters, records)

    return chromosome


def get_closest_records_cluster(chromosome, small_clusters, big_cluster_index, records):
    small_cluster_values = []
    small_cluster_records = []
    # print("small:", small_clusters)
    for cluster in small_clusters:
        # cluster = np.int64(cluster)
        # indices = np.where(chromosome == cluster)[0]
        indices = [index for index, elem in enumerate(chromosome) if elem == cluster]
        if len(indices) == 0:
            indices.append(big_cluster_index)
        # print("indices2 ", indices)
        for idx in indices:
            small_cluster_values.append(cluster)
            small_cluster_records.append(records[idx])
    small_cluster_records = np.array(small_cluster_records)
    distances = np.linalg.norm(small_cluster_records - records[big_cluster_index], axis=1)
    closest_index = np.argmin(distances)
    return small_cluster_values[closest_index]


# only one cluster should have size bigger than k
def rearrange_big_clusters_distance_based(chromosome, k, cluster_counts, big_clusters, records):
    if len(big_clusters) > 1:
        idx = get_cluster_with_smallest_sse(chromosome, big_clusters, records)
        chosen_cluster = big_clusters[idx]
        big_clusters.remove(chosen_cluster)

        for big_cluster in big_clusters:
            j = 0
            indices = arrange_records_in_cluster(chosen_cluster, big_cluster, chromosome, records)
            # print("arranged indices:", indices)
            count = 0
            while cluster_counts[big_cluster] > k:
                # print("indices[j] eseten j =", j)
                chromosome[indices[j]] = chosen_cluster
                j += 1
                cluster_counts[big_cluster] -= 1
                cluster_counts[chosen_cluster] += 1
                count += 1
                if (count > 100):
                    print("current cluster:", big_cluster)
                    print("cluster counts[big_cls]: ", cluster_counts[big_cluster])
                    print("indices: ", indices)
                    break
    return chromosome


def get_cluster_with_smallest_sse(chromosome, big_clusters, records):
    sse = []
    for i in big_clusters:
        cluster = records[chromosome == i]
        centroid = cluster.mean(axis=0)
        sse.append(np.sum((cluster - centroid) ** 2))

    sse = np.array(sse)
    return np.argmin(sse)


def arrange_records_in_cluster(chosen_cluster, big_cluster, chromosome, records):
    # chosen_cluster_records = records[chromosome == chosen_cluster]
    chosen_cluster_records = [records[g] for g, gene in enumerate(chromosome) if gene == chosen_cluster]
    # big_cluster_records = records[chromosome == big_cluster]
    big_cluster_records = [records[g] for g, gene in enumerate(chromosome) if gene == big_cluster]
    centroid = np.average(chosen_cluster_records, axis=0)
    distances = np.linalg.norm(big_cluster_records - centroid, axis=1)
    indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]

    paired = list(zip(distances, indices))
    paired.sort(key=lambda x: x[0])

    return [ind for _, ind in paired[:]]


def mutate_distance_based(chromosome, records, curr_iteration, max_iteration):
    start = 0.5
    end = 0.01
    mutation_rate = start - ((start - end) / max_iteration) * curr_iteration
    # for i in range(len(chromosome)):
    if random.random() < mutation_rate:
        i = random.randint(0, len(records) - 1)
        distances = np.linalg.norm(records - records[i], axis=1)
        closest_index = np.argmin(distances)
        swap_index = closest_index
        preferred_indices = np.where(chromosome == chromosome[closest_index])[0]
        while swap_index == closest_index:
            swap_index = np.random.choice(preferred_indices)

        aux = chromosome[i]
        chromosome[i] = chromosome[swap_index]
        chromosome[swap_index] = aux

    return chromosome


def mutate_distance_based2(chromosome, records, curr_iteration, max_iteration):
    start = 0.5
    end = 0.01
    mutation_rate = start - ((start - end) / max_iteration) * curr_iteration

    if random.random() < mutation_rate:
        # Kiszamoljuk minden klaszter centroidjat
        unique_clusters = np.unique(chromosome)
        centroids = {}

        for cluster in unique_clusters:
            cluster_indices = np.where(chromosome == cluster)[0]
            cluster_records = records[cluster_indices]
            centroids[cluster] = np.mean(cluster_records, axis=0)

        # Meghat. azt a rekordot amely legtavolabbik a centroidjatol
        farthest_distances = []
        for i, cluster in enumerate(chromosome):
            centroid = centroids[cluster]
            distance = np.linalg.norm(records[i] - centroid)
            farthest_distances.append((distance, i))

        farthest_distances.sort(reverse=True)
        farthest_index = farthest_distances[0][1]                           # ez lesz a gen ami felcserelodik

        # Meghat. a klasztert amibe athelyezzuk
        farthest_record = records[farthest_index]
        closest_cluster = min(
            unique_clusters,
            key=lambda cluster: np.linalg.norm(farthest_record - centroids[cluster]) if cluster != chromosome[
                farthest_index] else np.inf
        )

        # Meghat. az adott klaszter legrosszabb/legkulsobb elemet (ezzel csereljuk fel)
        closest_cluster_indices = np.where(chromosome == closest_cluster)[0]
        worst_index = max(
            closest_cluster_indices,
            key=lambda idx: np.linalg.norm(records[idx] - centroids[closest_cluster])
        )

        # Csere
        aux = chromosome[farthest_index]
        chromosome[farthest_index] = chromosome[worst_index]
        chromosome[worst_index] = aux

    return chromosome




def boosted_genetic_algorithm(records, n_clusters, generations, k, population_size, population):
    max_stagnation = 50
    shuffle_percentage = 0.3
    fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
    best_solution = population[np.argmin(fitnesses)]
    best_fitness = min(fitnesses)
    stagnation_count = 0

    for generation in range(generations):
        fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= max_stagnation:
            fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
            population = shuffle_random_population(population, fitnesses, shuffle_percentage)
            stagnation_count = 0
        parent_number = get_parent_number(population_size)
        # select best parents to crossover
        parents = tournament_selection(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover_distance_based(parent1, parent2, k, records)
            if child1 is not None and child2 is not None:
                offspring.append(mutate_distance_based2(child1, records, generation, generations))
                offspring.append(mutate_distance_based2(child2, records, generation, generations))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
        population = elitism_selection(population, new_fitnesses, population_size)

        if generation % 100 == 0:
            print(generation, ". generation:")
            print("Generations fitnesses:", fitnesses)
            print("Current best fitness:", best_fitness)

    calculate_inf_loss.calculate_I_loss(records, best_solution)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None

    dt_barcelona = '../Datasets/barcelona.csv'
    dt_Census = '../Datasets/Census.csv'
    dt_EIA = '../Datasets/EIA.csv'
    dt_madrid = '../Datasets/madrid.csv'
    dt_tarraco = '../Datasets/tarraco.csv'
    dt_tarragona = '../Datasets/tarragona.csv'
    datasets1 = [dt_barcelona,dt_madrid,dt_tarraco]
    datasets2 = [dt_EIA, dt_Census, dt_tarragona]
    # for dt in datasets1:
    #     print("*** WORKING ON:", dt)
    #     records = calculate_inf_loss.read_dataset_wo_header(dt)
    #     kx = [3, 4, 5]
    #     for k in kx:
    #         print("k=", k)
    #         for i in range(1):
    #             print(i, ". iteration ")
    #             population_size = 35
    #             generations = 500
    #             n_clusters = len(records) // k
    #             n_samples = records.shape[0]
    #             population = initialize_population(population_size, n_samples, n_clusters, k, records)
    #             best_solution, best_fitness = boosted_genetic_algorithm(records, n_clusters, generations, k,
    #                                                                     population_size, population)
    #             # best_solution, best_fitness = genetic_algorithm(records, n_clusters, generations, k, population_size,
    #             #                                                 population)
    #             print("Best Fitness (SSE):", best_fitness)
    #
    #             overall_mean = np.mean(records, axis=0)
    #             SST = np.sum((records - overall_mean) ** 2)
    #             print("I= ", (best_fitness / SST) * 100)

    for dt in datasets2:
        print("*** WORKING ON:", dt)
        records = calculate_inf_loss.read_dataset(dt)
        kx = [3, 4, 5]
        for k in kx:
            print("k=", k)
            for i in range(1):
                print(i, ". iteration ")
                population_size = 35
                generations = 500
                n_clusters = len(records) // k
                n_samples = records.shape[0]
                population = initialize_population(population_size, n_samples, n_clusters, k, records)
                best_solution, best_fitness = boosted_genetic_algorithm(records, n_clusters, generations, k,
                                                                        population_size, population)
                # best_solution, best_fitness = genetic_algorithm(records, n_clusters, generations, k, population_size,
                #                                                 population)
                print("Best Fitness (SSE):", best_fitness)

                overall_mean = np.mean(records, axis=0)
                SST = np.sum((records - overall_mean) ** 2)
                print("I= ", (best_fitness / SST) * 100)


if __name__ == "__main__":
    main()

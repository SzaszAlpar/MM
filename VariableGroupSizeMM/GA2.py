import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
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


def initialize_population(size, chromosome_length, min_clusters, max_clusters, k_min):
    population = []
    for _ in range(size):
        n_clusters = random.randint(min_clusters, max_clusters)
        chromosome = np.zeros(chromosome_length, dtype=int)
        cluster_sizes = np.random.randint(k_min, chromosome_length // n_clusters + 1, size=n_clusters)
        remaining = chromosome_length - cluster_sizes.sum()

        for i in range(remaining):
            cluster_sizes[random.randint(0, n_clusters - 1)] += 1

        start = 0
        for cluster_num in range(n_clusters):
            end = start + cluster_sizes[cluster_num]
            chromosome[start:end] = cluster_num
            start = end

        np.random.shuffle(chromosome)
        population.append((chromosome, n_clusters))

    return population


def fitness(chromosome_tuple, data):
    chromosome, n_clusters = chromosome_tuple
    sse = 0
    for i in range(n_clusters):
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


def select_in_elite_mode(population, fitnesses, number_of_chromosomes):
    probabilities = 1 / np.array(fitnesses)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(population), size=number_of_chromosomes, p=probabilities)
    selected_pop = [population[i] for i in selected_indices]
    selected_fit = [fitnesses[i] for i in selected_indices]

    selected_pop = [x for _, x in sorted(zip(selected_fit, selected_pop))]
    return selected_pop


def uniform_crossover(parent1_tuple, parent2_tuple, crossover_rate=0.80, parent_rate=0.5):
    parent1, n_clusters1 = parent1_tuple
    parent2, n_clusters2 = parent2_tuple
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
        return (np.array(child1), n_clusters1), (np.array(child2), n_clusters2)
    return parent1_tuple, parent2_tuple


def uniform_crossover2(parent1_tuple, parent2_tuple, k, crossover_rate=0.80, parent_rate=0.5):
    parent1, n_clusters1 = parent1_tuple
    parent2, n_clusters2 = parent2_tuple

    if n_clusters1 <= n_clusters2:
        cluster_differences = list(range(n_clusters1, n_clusters2))
    else:
        cluster_differences = list(range(n_clusters2, n_clusters1))
        parent1, parent2 = parent2, parent1
        n_clusters1, n_clusters2 = n_clusters2, n_clusters1

    if random.random() < crossover_rate:
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            random_nr = random.random()
            if (parent1[i] in cluster_differences) or (random_nr < parent_rate):
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        child1 = rearrange_chromosome(child1, k)
        child2 = rearrange_chromosome(child2, k)
        return (np.array(child1), n_clusters1), (np.array(child2), n_clusters2)
    return parent1_tuple, parent2_tuple


def rearrange_chromosome(chromosome, k):
    cluster_counts = {}
    for gene in chromosome:
        if gene in cluster_counts:
            cluster_counts[gene] += 1
        else:
            cluster_counts[gene] = 1

    small_clusters = [cluster for cluster, count in cluster_counts.items() if count < k]
    big_clusters = [cluster for cluster, count in cluster_counts.items() if count > k]

    for cluster in small_clusters:
        while cluster_counts[cluster] < k:
            big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])
            indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]
            random_index = random.choice(indices)
            chromosome[random_index] = cluster

            cluster_counts[cluster] += 1
            cluster_counts[big_cluster] -= 1
    return chromosome


def mutate(chromosome_tuple, mutation_rate=0.1):
    chromosome, n_clusters = chromosome_tuple
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            rand_index = random.randint(0, chromosome.size - 1)
            aux = chromosome[i]
            chromosome[i] = chromosome[rand_index]
            chromosome[rand_index] = aux

    return (chromosome, n_clusters)


def get_parent_number(population, parent_rate=0.60):
    number = int(population * parent_rate)
    if number % 2 != 0:
        return number + 1
    else:
        return number


def genetic_algorithm(records, min_clusters, max_clusters, generations, k_min, population_size):
    n_samples = records.shape[0]
    population = initialize_population(population_size, n_samples, min_clusters, max_clusters, k_min)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        print(generation, ". generation:")
        fitnesses = [fitness(chrom_tuple, records) for chrom_tuple in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
        parent_number = get_parent_number(population_size)
        parents = select_in_elite_mode(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover2(parent1, parent2,k_min)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom_tuple, records) for chrom_tuple in population]
        population = select_in_elite_mode(population, new_fitnesses, population_size)

        print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    [records, sc, full_data] = read_data_normalized()
    k_min = 25
    population_size = 43
    generations = 20
    min_clusters = 10
    max_clusters = len(records) // k_min

    best_solution, best_fitness = genetic_algorithm(records, min_clusters, max_clusters, generations, k_min,
                                                    population_size)
    print("Best Solution:", best_solution)
    print("Best Fitness (SSE):", best_fitness)

    interpret_result(best_solution, full_data, records, sc)


def interpret_result(best_solution, full_data, records, sc):
    best_chromosome, best_n_clusters = best_solution
    centroids = ResultInterpreter.aggregate(records, best_chromosome)
    groups = ResultInterpreter.get_groups(records, best_chromosome)
    for group in groups:
        print("group len:", len(group))
    RI = ResultInterpreter.Interpreter(groups, centroids, sc)
    RI.set_full_groups(full_data, best_chromosome)
    RI.print_group_analysis([3, 8, 2])
    RI.plot_two_column_of_centroids('Quality of Sleep', 'Stress Level')
    RI.plot_two_column_of_centroids('Physical Activity Level', 'Daily Steps')
    RI.calculate_homogeneity()


def main2():
    k = 2
    chromosome = [1, 4, 2, 1, 2, 3, 4, 4, 2, 5, 5, 4, 6]
    new_chromosome = rearrange_chromosome(chromosome, k)
    print(new_chromosome)


if __name__ == "__main__":
    main()

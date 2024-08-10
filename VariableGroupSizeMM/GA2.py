import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
from FixedGroupSizeMM import ResultInterpreter, SA


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


def initialize_population(size, chromosome_length, k, records):
    population = []
    min_clusters = chromosome_length // (2 * k - 1)
    max_clusters = chromosome_length // k
    for _ in range(size):
        n_clusters = random.randint(min_clusters, max_clusters)
        chromosome = np.zeros(chromosome_length, dtype=int)
        cluster_sizes = k * np.ones(n_clusters, dtype=int)
        remaining = chromosome_length - cluster_sizes.sum()

        for i in range(remaining):
            cluster_sizes[random.randint(0, n_clusters - 1)] += 1

        start = 0
        for cluster_num in range(n_clusters):
            end = start + cluster_sizes[cluster_num]
            chromosome[start:end] = cluster_num
            start = end

        np.random.shuffle(chromosome)
        print("chromosome fitness:", fitness(chromosome, records))
        population.append(chromosome)

    return population


def initialize_population_and_adjust_with_SA(size, chromosome_length, k, records):
    pop = initialize_population(size, chromosome_length, k, records)
    initial_temperature = 80
    cooling_rate = 0.85
    max_iterations = 200
    min_energy_threshold = 1e-5
    max_stagnation_iterations = 25
    new_pop = []
    for ch in pop:
        new_ch, fitness = SA.simulated_annealing22(records, k, initial_temperature, cooling_rate, max_iterations,
                                                   min_energy_threshold, max_stagnation_iterations, ch)
        print("fitness", fitness)
        new_pop.append(new_ch)
    return new_pop


def fitness(chromosome, data):
    sse = 0
    n_clusters = np.unique(chromosome)
    for i in n_clusters:
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


def tournament_selection(population, fitnesses, number_of_chromosomes, tournament_size=3):
    selected_pop = []

    for _ in range(number_of_chromosomes):
        # Randomly select individuals for the tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_pop = [population[i] for i in tournament_indices]
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Choose the best individual from the tournament
        best_index = np.argmin(tournament_fitnesses)
        selected_pop.append(tournament_pop[best_index])

    return selected_pop


def uniform_crossover(parent1, parent2, k, crossover_rate=0.80, parent_rate=0.5):
    n_clusters1 = np.unique(parent1)
    n_clusters2 = np.unique(parent2)

    if len(n_clusters1) <= len(n_clusters2):
        cluster_differences = list(range(len(n_clusters1), len(n_clusters2)))
    else:
        cluster_differences = list(range(len(n_clusters2), len(n_clusters1)))
        parent1, parent2 = parent2, parent1

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
        return np.array(child1), np.array(child2)
    return parent1, parent2


def rearrange_chromosome(chromosome, k):
    cluster_counts = {}
    for gene in chromosome:
        if gene in cluster_counts:
            cluster_counts[gene] += 1
        else:
            cluster_counts[gene] = 1

    small_clusters = [cluster for cluster, count in cluster_counts.items() if count < k]
    big_clusters = [cluster for cluster, count in cluster_counts.items() if count > k]
    big_cluster = max(big_clusters, key=lambda x: cluster_counts[x])
    while small_clusters:
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
    return chromosome


def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            rand_index = random.randint(0, chromosome.size - 1)
            aux = chromosome[i]
            chromosome[i] = chromosome[rand_index]
            chromosome[rand_index] = aux

    return chromosome


def get_parent_number(population, parent_rate=0.60):
    number = int(population * parent_rate)
    if number % 2 != 0:
        return number + 1
    else:
        return number


def genetic_algorithm(records, generations, k, population_size):
    n_samples = records.shape[0]
    population = initialize_population_and_adjust_with_SA(population_size, n_samples, k, records)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        print(generation, ". generation:")
        fitnesses = [fitness(chrom, records) for chrom in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
        parent_number = get_parent_number(population_size)
        parents = tournament_selection(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover(parent1, parent2, k)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records) for chrom in population]
        population = tournament_selection(population, new_fitnesses, population_size)
        population.append(best_solution)

        print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    [records, sc, full_data] = read_data_normalized()
    k = 3
    population_size = 10
    generations = 220

    best_solution, best_fitness = genetic_algorithm(records, generations, k, population_size)
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


if __name__ == "__main__":
    main()

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


def initialize_population(size, chromosome_length, n_clusters, k):
    population = []
    for _ in range(size):
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


def fitness(chromosome, data, n_clusters):
    sse = 0
    for i in range(n_clusters):
        # we want to get all data points assigned to cluster i
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


def select_in_elite_mode(population, fitnesses, number_of_chromosomes):
    # Convert fitnesses to probabilities for selection (inverse since lower fitness is better)
    probabilities = 1 / np.array(fitnesses)
    # normalizing it so the sum became 1
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(len(population), size=number_of_chromosomes, p=probabilities)
    selected_pop = [population[i] for i in selected_indices]
    selected_fit = [fitnesses[i] for i in selected_indices]

    # sort by fitness value
    # selected_pop = [x for _, x in sorted(zip(selected_fit, selected_pop))]
    selected_pop = [x for _, x in sorted(zip(selected_fit, selected_pop), key=lambda pair: pair[0].sum())]

    return selected_pop


def crossover(parent1, parent2, crossover_rate=0.80):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 2)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2


def uniform_crossover(parent1, parent2, k, crossover_rate=0.80, parent_rate=0.5):
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

    while len(big_clusters) > 1 and len(small_clusters) > 0:
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
        chromosome = rearrange_chromosome2(chromosome, k, cluster_counts, big_clusters)

    return chromosome


def rearrange_chromosome2(chromosome, k, cluster_counts, big_clusters):
    if len(big_clusters) > 1:
        chosen_cluster = random.choice(big_clusters)
        big_clusters.remove(chosen_cluster)

        for big_cluster in big_clusters:
            while cluster_counts[big_cluster] > k:
                indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]
                random_index = random.choice(indices)
                chromosome[random_index] = chosen_cluster
                cluster_counts[big_cluster] -= 1
                cluster_counts[chosen_cluster] += 1
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


def genetic_algorithm(records, n_clusters, generations, k, population_size):
    n_samples = records.shape[0]
    population = initialize_population(population_size, n_samples, n_clusters, k)
    best_solution = None
    best_fitness = float('inf')

    for generation in range(generations):
        # print(generation, ". generation:")
        fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
        parent_number = get_parent_number(population_size)
        # select best parents to crossover
        parents = select_in_elite_mode(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover(parent1, parent2, k)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
        population = select_in_elite_mode(population, new_fitnesses, population_size)

        # print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None
    [records, sc, full_data] = read_data_normalized()
    k = 10
    population_zise = 76
    generations = 10
    n_clusters = len(records) // k

    best_solution, best_fitness = genetic_algorithm(records, n_clusters, generations, k, population_zise)
    print("Best Solution:", best_solution)
    print("Best Fitness (SSE):", best_fitness)

    interpret_result(best_solution, full_data, records, sc)


def interpret_result(best_solution, full_data, records, sc):
    centroids = ResultInterpreter.aggregate(records, best_solution)
    groups = ResultInterpreter.get_groups(records, best_solution)
    for group in groups:
        print("group len:", len(group))
    RI = ResultInterpreter.Interpreter(groups, centroids, sc)
    RI.set_full_groups(full_data, best_solution)
    RI.print_group_analysis([3, 8, 2])
    RI.plot_two_column_of_centroids('Quality of Sleep', 'Stress Level')
    RI.plot_two_column_of_centroids('Physical Activity Level', 'Daily Steps')
    RI.calculate_homogeneity()


if __name__ == "__main__":
    main()

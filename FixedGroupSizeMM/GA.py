import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import ResultInterpreter
import calculate_inf_loss, SA
import time


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

        print("chromosome fitness:", fitness(chromosome, records, n_clusters))
        population.append(chromosome)

    return population


def initialize_population_and_adjust_with_SA(size, chromosome_length, n_clusters, k, records):
    pop = initialize_population(size, chromosome_length, n_clusters, k, records)
    initial_temperature = 100
    cooling_rate = 0.85
    max_iterations = 400
    min_energy_threshold = 1e-5
    max_stagnation_iterations = 25
    new_pop = []
    for ch in pop:
        new_ch, fitness = SA.simulated_annealing22(records, k, initial_temperature, cooling_rate, max_iterations,
                                                   min_energy_threshold, max_stagnation_iterations, ch)
        print("fitness", fitness)
        new_pop.append(new_ch)
    return new_pop


def fitness(chromosome, data, n_clusters):
    sse = 0
    for i in range(n_clusters):
        # we want to get all data points assigned to cluster i
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


def elitism_selection(population, fitnesses, number_of_chromosomes):
    paired = list(zip(fitnesses, population))
    # Sort the population by fitness
    paired.sort(key=lambda x: x[0])
    selected_pop = [ind for _, ind in paired[:number_of_chromosomes]]
    return selected_pop


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


def uniform_crossover(parent1, parent2, k, crossover_rate=0.90, parent_rate=0.5):
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


# only one cluster should have size bigger than k
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


def mutate(chromosome, mutation_rate=0.01):
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


def shuffle_top_population(population, fitnesses, shuffle_percentage):
    paired = list(zip(fitnesses, population))
    paired.sort(key=lambda x: x[0])

    sorted_population = [ind for _, ind in paired[:]]
    num_to_shuffle = int(len(sorted_population) * shuffle_percentage)
    for idx in range(num_to_shuffle):
        np.random.shuffle(sorted_population[idx])

    return sorted_population


def genetic_algorithm(records, n_clusters, generations, k, population_size):
    n_samples = records.shape[0]
    max_stagnation = 30
    shuffle_percentage = 0.5
    population = initialize_population_and_adjust_with_SA(population_size, n_samples, n_clusters, k, records)
    fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
    best_solution = population[np.argmin(fitnesses)]
    best_fitness = min(fitnesses)
    stagnation_count = 0

    for generation in range(generations):
        print(generation, ". generation:")
        fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]

        if min(fitnesses) < best_fitness:
            best_fitness = min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= max_stagnation:
            fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
            population = shuffle_top_population(population, fitnesses, shuffle_percentage)
            stagnation_count = 0
        parent_number = get_parent_number(population_size)
        # select best parents to crossover
        parents = tournament_selection(population, fitnesses, parent_number)
        offspring = []

        for i in range(0, parent_number, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = uniform_crossover(parent1, parent2, k)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
        population = tournament_selection(population, new_fitnesses, population_size)
        population.append(best_solution)

        print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None
    # [records, sc, full_data] = read_data_normalized()
    dataset_Census = '../Datasets/Census.csv'
    records = calculate_inf_loss.read_dataset(dataset_Census)
    k = 3
    population_zise = 10
    generations = 100
    n_clusters = len(records) // k

    best_solution, best_fitness = genetic_algorithm(records, n_clusters, generations, k, population_zise)
    # print("Best Solution:", best_solution)
    print("Best Fitness (SSE):", best_fitness)

    # interpret_result(best_solution, full_data, records, sc)
    overall_mean = np.mean(records, axis=0)
    SST = np.sum((records - overall_mean) ** 2)
    print("I= ", (best_fitness / SST) * 100)


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

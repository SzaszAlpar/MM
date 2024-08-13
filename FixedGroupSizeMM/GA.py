import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
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
    initial_temperature = 80
    cooling_rate = 0.85
    max_iterations = 500
    min_energy_threshold = 1e-5
    max_stagnation_iterations = 25
    first_half = size // 2
    new_pop = []
    for ch in pop[:first_half]:
        new_ch, fitness = SA.simulated_annealing22(records, k, initial_temperature, cooling_rate, max_iterations,
                                                   min_energy_threshold, max_stagnation_iterations, ch)
        print("fitness", fitness)
        new_pop.append(new_ch)
    return new_pop + pop[first_half + 1:]


def fitness(chromosome, data, n_clusters):
    sse = 0
    for i in range(n_clusters):
        # we want to get all data points assigned to cluster i
        cluster = data[chromosome == i]
        if len(cluster) > 0:
            centroid = cluster.mean(axis=0)
            sse += np.sum((cluster - centroid) ** 2)
    return sse


# half of the chosen population is picked by elite mode
def elitism_selection(population, fitnesses, number_of_chromosomes):
    paired = list(zip(fitnesses, population))
    first_half = number_of_chromosomes // 3
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


def uniform_crossover(parent1, parent2, k, records, crossover_rate=0.90, parent_rate=0.5):
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
        child1 = rearrange_chromosome(child1, k, records)
        child2 = rearrange_chromosome(child2, k, records)
        return np.array(child1), np.array(child2)
    return None, None


def rearrange_chromosome(chromosome, k, records):
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
            # small_cluster = small_clusters[0]
            indices = np.where(chromosome == big_cluster)[0]
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
        chromosome = rearrange_chromosome2(chromosome, k, cluster_counts, big_clusters, records)

    return chromosome


def get_closest_records_cluster(chromosome, small_clusters, big_cluster_index, records):
    small_cluster_values = []
    small_cluster_records = []
    for cluster in small_clusters:
        indices = np.where(chromosome == cluster)[0]
        for idx in indices:
            small_cluster_values.append(cluster)
            small_cluster_records.append(records[idx])
    small_cluster_records = np.array(small_cluster_records)
    distances = np.linalg.norm(small_cluster_records - records[big_cluster_index], axis=1)
    closest_index = np.argmin(distances)
    return small_cluster_values[closest_index]


# only one cluster should have size bigger than k
def rearrange_chromosome2(chromosome, k, cluster_counts, big_clusters, records):
    if len(big_clusters) > 1:
        idx = get_cluster_with_smallest_sse(chromosome, big_clusters, records)
        chosen_cluster = big_clusters[idx]
        big_clusters.remove(chosen_cluster)

        for big_cluster in big_clusters:
            j = 0
            indices = arrange_records_in_cluster(chosen_cluster, big_cluster, chromosome, records)
            while cluster_counts[big_cluster] > k:
                chromosome[indices[j]] = chosen_cluster
                j += 1
                cluster_counts[big_cluster] -= 1
                cluster_counts[chosen_cluster] += 1
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
    chosen_cluster_records = records[chromosome == chosen_cluster]
    big_cluster_records = records[chromosome == big_cluster]
    centroid = np.average(chosen_cluster_records, axis=0)
    distances = np.linalg.norm(big_cluster_records - centroid, axis=1)
    indices = [i for i, gene in enumerate(chromosome) if gene == big_cluster]

    paired = list(zip(distances, indices))
    paired.sort(key=lambda x: x[0])

    return [ind for _, ind in paired[:]]


def mutate(chromosome, mutation_rate=0.01):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            rand_index = random.randint(0, chromosome.size - 1)
            aux = chromosome[i]
            chromosome[i] = chromosome[rand_index]
            chromosome[rand_index] = aux

    return chromosome


def mutate2(chromosome, records, curr_iteration, max_iteration):
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
    max_stagnation = 50
    shuffle_percentage = 0.2
    population = initialize_population(population_size, n_samples, n_clusters, k, records)
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
            child1, child2 = uniform_crossover(parent1, parent2, k, records)
            if child1 is not None and child2 is not None:
                offspring.append(mutate2(child1, records, generation, generations))
                offspring.append(mutate2(child2, records, generation, generations))

        population.extend(offspring)
        new_fitnesses = [fitness(chrom, records, n_clusters) for chrom in population]
        print("Generations fitnesses:", fitnesses)
        population = elitism_selection(population, new_fitnesses, population_size)
        # population.append(best_solution)

        print("Current best fitness:", best_fitness)

    return best_solution, best_fitness


def main():
    pd.options.mode.chained_assignment = None
    # [records, sc, full_data] = read_data_normalized()
    dataset_Census = '../Datasets/Census.csv'
    records = calculate_inf_loss.read_dataset(dataset_Census)
    k = 3
    population_zise = 50
    generations = 500
    n_clusters = len(records) // k

    best_solution, best_fitness = genetic_algorithm(records, n_clusters, generations, k, population_zise)
    # print("Best Solution:", best_solution)
    print("Best Fitness (SSE):", best_fitness)

    # interpret_result(best_solution, full_data, records, sc)
    overall_mean = np.mean(records, axis=0)
    SST = np.sum((records - overall_mean) ** 2)
    print("I= ", (best_fitness / SST) * 100)


def main2():
    start = 0.05
    end = 0.01
    max_iteration = 500
    for curr_iteration in range(max_iteration):
        mutation_rate = start - ((start - end) / max_iteration) * curr_iteration
        print(mutation_rate)


if __name__ == "__main__":
    main()

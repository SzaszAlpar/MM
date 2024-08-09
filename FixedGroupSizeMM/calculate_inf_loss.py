import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import MDAV, Kmean, GA, SA, PSO, MDAV_GA
from VariableGroupSizeMM import VMDAV, VKmean, GA2, SA2, PSO2, Graphh, MDAV_graph
import sys
import os
from collections import defaultdict
from scipy.spatial import distance_matrix


def suppress_output():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def restore_output():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def read_dataset_wo_header(path):
    df = pd.read_csv(path, sep=';', header=None)

    scalers = {}
    for column in range(len(df.columns)):
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler
    df = df.fillna(0).to_numpy()
    return df


def read_dataset(path):
    df = pd.read_csv(path)
    columns = df.columns
    scalers = {}
    for column in columns:
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[column].to_numpy().reshape(-1, 1))
        scalers[column] = scaler
    df = df.fillna(0).to_numpy()
    return df

def run_MDAV(kx, records):
    for k in kx:
        suppress_output()
        groups, indices = MDAV.MDAV(records, k)
        restore_output()

        cluster_assignment = np.zeros(len(records), dtype=int)

        group_lengths = [len(gr) for gr in indices]
        print("MDAV, k=", k, ".\nGroups have the following sizes: ", group_lengths)
        for cluster_id, record_indices in enumerate(indices):
            for record_index in record_indices:
                cluster_assignment[record_index] = cluster_id

        calculate_I_loss(records, cluster_assignment)


def run_VMDAV(kx, records):
    for k in kx:
        suppress_output()
        groups, indices = VMDAV.V_MDAV(records, k)
        restore_output()

        cluster_assignment = np.zeros(len(records), dtype=int)

        group_lengths = [len(gr) for gr in indices]
        print("V-MDAV, k=", k, ".\nGroups have the following sizes: ", group_lengths)
        for cluster_id, record_indices in enumerate(indices):
            for record_index in record_indices:
                cluster_assignment[record_index] = cluster_id

        calculate_I_loss(records, cluster_assignment)


def run_KMean(kx, records):
    for k in kx:
        suppress_output()
        groups, cluster_assignment = Kmean.kmeans_microaggregation(records, k)
        restore_output()

        group_lengths = [len(gr) for gr in groups]
        print("Kmean, k=", k, ".\nGroups have the following sizes: ", group_lengths)

        calculate_I_loss(records, cluster_assignment)


def run_VKMean(kx, records):
    for k in kx:
        suppress_output()
        groups, cluster_assignment = VKmean.kmeans_microaggregation(records, k)
        restore_output()

        group_lengths = [len(gr) for gr in groups]
        print("V-Kmean, k=", k, ".\nGroups have the following sizes: ", group_lengths)

        calculate_I_loss(records, cluster_assignment)


def run_GA(kx, records):
    for k in kx:
        suppress_output()
        population_zise = 50
        generations = 100
        n_clusters = len(records) // k

        cluster_assignment, best_fitness = GA.genetic_algorithm(records, n_clusters, generations, k, population_zise)
        restore_output()

        print("GA, k=", k, ".\nReturned best fitness: ", best_fitness)

        calculate_I_loss(records, cluster_assignment)


# Ezt az algoritmust kell modositani, nem talal a visszateritatt fitness a kiszamolt SSE-vel (ugyanaz kellene legyen)
def run_VGA(kx, records):
    for k in kx:
        suppress_output()
        population_zise = 50
        generations = 3
        max_clusters = len(records) // k
        min_clusters = len(records) // (2 * k - 1)

        best_solution, best_fitness = GA2.genetic_algorithm(records, min_clusters, max_clusters, generations, k,
                                                            population_zise)
        cluster_assignment, best_n_clusters = best_solution
        restore_output()

        print("VGA, k=", k, ".\nReturned best fitness: ", best_fitness, " , result has ", best_n_clusters, " clusters")

        calculate_I_loss(records, cluster_assignment)


def run_SA(kx, records):
    initial_temperature = 1000
    cooling_rate = 0.99
    max_iterations = 2000
    min_energy_threshold = 1e-5
    max_stagnation_iterations = 100
    for k in kx:
        suppress_output()
        cluster_assignment, best_energy = SA.simulated_annealing2(records, k, initial_temperature, cooling_rate,
                                                                  max_iterations, min_energy_threshold,
                                                                  max_stagnation_iterations)
        restore_output()

        print("SA, k=", k, ".\nReturned best fitness: ", best_energy)

        calculate_I_loss(records, cluster_assignment)


def run_VSA(kx, records):
    initial_temperature = 1000
    cooling_rate = 0.99
    max_iterations = 2000
    min_energy_threshold = 1e-5
    max_stagnation_iterations = 100
    for k in kx:
        suppress_output()
        k_min = k
        k_max = 2 * k - 1
        cluster_assignment, best_energy = SA2.simulated_annealing2(records, k_min, k_max, initial_temperature,
                                                                   cooling_rate,
                                                                   max_iterations, min_energy_threshold,
                                                                   max_stagnation_iterations)
        restore_output()

        print("VSA, k=", k, ".\nReturned best fitness: ", best_energy)

        calculate_I_loss(records, cluster_assignment)


def run_PSO(kx, records):
    num_particles = 15
    max_iterations = 100
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    for k in kx:
        suppress_output()
        [gBest, gBest_value] = PSO.PSO(records, k, num_particles, max_iterations, w, c1, c2)
        cluster_assignment = PSO.assign_data_to_clusters(records, gBest, k)
        restore_output()

        print("PSO, k=", k, ".\nReturned best fitness: ", gBest_value)

        calculate_I_loss(records, cluster_assignment)


def run_VPSO(kx, records):
    num_particles = 15
    max_iterations = 100
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    for k in kx:
        suppress_output()
        [gBest, gBest_value] = PSO2.PSO(records, k, num_particles, max_iterations, w, c1, c2)
        cluster_assignment = PSO2.assign_data_to_clusters(records, gBest, k)
        restore_output()

        print("VPSO, k=", k, ".\nReturned best fitness: ", gBest_value)

        calculate_I_loss(records, cluster_assignment)


def run_GRAPH(kx, records):
    for k in kx:

        n = len(records)
        adjacency_list = defaultdict(list)
        parents = [-1] * n
        dm = distance_matrix(records, records)
        suppress_output()
        groups = Graphh.run(records, n, k, adjacency_list, parents, dm)
        cluster_assignment = np.zeros(len(records), dtype=int)
        group_lengths = [len(gr) for gr in groups]
        for cluster_id, record_indices in enumerate(groups):
            for record_index in record_indices:
                cluster_assignment[record_index] = cluster_id

        restore_output()

        print("GRAPH, k=", k, ".\nReturned group's length: ", group_lengths)

        calculate_I_loss(records, cluster_assignment)


def run_MDAV_GRAPH(kx, records):
    for k in kx:
        suppress_output()
        cluster_assignment = MDAV_graph.MDAV_Graph(records, 50, k)
        restore_output()

        print("MDAV+GRAPH, k=", k)

        calculate_I_loss(records, cluster_assignment)


def run_MDAV_GA(kx, records):
    for k in kx:
        suppress_output()
        population_size = 100
        generations = 100
        cluster_assignment = MDAV_GA.MDAV_GA(records, 30, k, population_size, generations)
        restore_output()

        print("MDAV+GA, k=", k)

        calculate_I_loss(records, cluster_assignment)


def calculate_I_loss(data, result):
    # SST
    overall_mean = np.mean(data, axis=0)
    SST = np.sum((data - overall_mean) ** 2)

    # SSE
    SSE = 0
    unique_groups = np.unique(result)
    for group in unique_groups:
        group_indices = np.where(result == group)
        group_data = data[group_indices]
        group_mean = np.mean(group_data, axis=0)
        SSE += np.sum((group_data - group_mean) ** 2)

    print("SST:", SST)
    print("SSE:", SSE)
    print("I_loss:", (SSE / SST) * 100)
    print("\n")


if __name__ == "__main__":
    dataset_barcelona = '../Datasets/barcelona.csv'
    dataset_CASCrefmicrodata = '../Datasets/CASCrefmicrodata.csv'
    # records = read_dataset_wo_header(dataset_barcelona)
    records = read_dataset(dataset_CASCrefmicrodata)
    k = [3, 4, 5, 6]
    run_MDAV(k,records)
    run_VMDAV(k,records)
    run_KMean(k,records)
    run_VKMean(k,records)
    # run_GA(k,records)
    # run_VGA(k,records)
    # run_SA(k,records)
    # run_VSA(k,records)
    # run_PSO(k,records)
    # run_VPSO(k,records)
    run_GRAPH(k,records)
    run_MDAV_GRAPH(k,records)
    run_MDAV_GA(k, records)

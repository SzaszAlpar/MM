from FixedGroupSizeMM import MDAV
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import GA


def read_data_normalized():
    df = pd.read_csv('../FixedGroupSizeMM/Sleep_health_and_lifestyle_dataset.csv')
    df = df[
        ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate',
         'Daily Steps', ]]

    column_names = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate',
                    'Daily Steps']
    scaler = StandardScaler()
    for columnName in column_names:
        scaled_data = scaler.fit_transform(df[columnName].to_numpy().reshape(-1, 1))
        df[columnName] = scaled_data

    df = df.fillna(0).to_numpy()
    return df


def main():
    records = read_data_normalized()
    n = len(records)
    k = 50
    k2 = 10
    final_assignments = -1 * np.ones(n, dtype=int)
    groups, indices = MDAV.MDAV(records, k)

    idx = 0

    for i, group in enumerate(groups):
        print(i, ". th group from MDAV has ", len(group), " samples. Call GA on it, k=", k2)
        k2 = 10
        population_zise = 60
        generations = 300
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

if __name__ == "__main__":
    main()

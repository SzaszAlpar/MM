import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
from collections import defaultdict, deque
from scipy.spatial import distance_matrix


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


def get_k_nearest_neighbors(u, k, distance_matrix):
    dists = distance_matrix[u]
    nearest_neighbors = np.argsort(dists)
    return nearest_neighbors[1:k + 1]


# Ez az algoritmus inkabb koveti a cikkben levo PSEUDOKODOT
# def forest(n, k, adjacency_list, parents, distance_matrix):
#     components = {i: {i} for i in range(n)}
#     components_ind = list(range(n))
#
#     while any(len(comp) < k for comp in components.values()):
#         small_components = {i for i, comp in components.items() if len(comp) < k}
#
#         for comp_id in small_components:
#             if len(components[comp_id]) >= k:
#                 continue
#             component = components[comp_id]
#             for u in component:
#                 if not adjacency_list[u]:
#                     nearest_neighbors = get_k_nearest_neighbors(u, k, distance_matrix)
#                     for v in nearest_neighbors:
#                         if v not in component:
#                             components[v].update(component)
#                             for w in component:
#                                 components[w] = components[v]
#                                 components_ind[w] = components_ind[v]
#                             adjacency_list[u].append(v)
#                             parents[v] = u
#                             break
#
#     components_ind = list(set(components_ind))
#     reduced_components = {i: components[j] for i, j in enumerate(components_ind)}
#     return reduced_components


def forest(n, k, adjacency_list, parents, distance_matrix):
    components = {i: {i} for i in range(n)}
    vertices = set(range(n))
    components_ind = list(range(n))

    while any(len(comp) < k for comp in components.values()):
        for u in vertices:
            if len(components[u]) >= k or u in adjacency_list:
                continue
            nearest_neighbors = get_k_nearest_neighbors(u, k, distance_matrix)
            for v in nearest_neighbors:
                if v not in components[u]:
                    components[v].update(components[u])
                    for w in components[u]:
                        components[w] = components[v]
                        components_ind[w] = components_ind[v]

                    adjacency_list[u].append(v)
                    parents[v] = u
                    break

    components_ind = list(set(components_ind))
    reduced_components = {i: components[j] for i, j in enumerate(components_ind)}
    return reduced_components


def compute_subtree_sizes(tree, node):
    subtree_sizes = {}

    def helper(node):
        size = 1
        for child in tree[node]:
            size += helper(child)
        subtree_sizes[node] = size
        return size

    helper(node)
    return subtree_sizes


def decompose_component(component, k, adjacency_list):
    max_size = max(2 * k - 1, 3 * k - 5)
    while len(component) > max_size:
        # Azt a csucsot valasszuk amelyikbe nem megy el
        vertexes = [num for num in component if num not in adjacency_list]
        print("vertexes:", vertexes)
        u = vertexes[0]

        # Step 3: Letrehozzuk a komponens-fat, majd a reszfak hosszat hatarozzuk meg
        tree = defaultdict(list)
        queue = deque([u])
        visited = set([u])

        while queue:
            node = queue.popleft()
            neighbours = [key for key, value_list in adjacency_list.items() if node in value_list]
            for neighbor in neighbours:
                if neighbor in component and neighbor not in visited:
                    tree[node].append(neighbor)
                    queue.append(neighbor)
                    visited.add(neighbor)

        subtree_sizes = compute_subtree_sizes(tree, u)
        del subtree_sizes[u]

        # Meghatarozzuk a legnagyobb reszfa gyokeret
        largest_subtree_size = max(subtree_sizes.values())
        largest_subtree_root = [key for key, value in subtree_sizes.items() if value == largest_subtree_size][0]

        # Szetvalasztjuk a fat meretek alapjan
        s = len(component)
        phi = largest_subtree_size
        print("s:", s, ", phi:", phi, ", k:", k)

        if s - phi >= k - 1:
            if phi >= k and s - phi >= k:
                # megtartsuk a legnagyobb reszfat, a tobbi egy masik csoport
                print("ELSO ESET")
                new_component_1 = [largest_subtree_root]
                queue = deque(tree[largest_subtree_root])
                while queue:
                    node = queue.popleft()
                    new_component_1.append(node)
                    queue.extend(tree[node])
                new_component_2 = [node for node in component if node not in new_component_1]
                component = new_component_1
                return [component, new_component_2]
            elif s - phi == k - 1:
                # a legnagyobb reszfabol levagjuk a gyokeret, ezt a tobbivel egyutt csoportositjuk
                print("MASODIK ESET")
                new_component_1 = []
                queue = deque(tree[largest_subtree_root])
                while queue:
                    node = queue.popleft()
                    new_component_1.append(node)
                    queue.extend(tree[node])
                new_component_2 = [node for node in component if node not in new_component_1]
                component = new_component_1
                return [component, new_component_2]
            elif phi == k - 1:
                # a legnagyobb reszfahoz hozzadjuk meg az alap fa gyokeret is
                print("HARMADIK ESET")
                new_component_1 = [u] + [largest_subtree_root]
                queue = deque(tree[largest_subtree_root])
                while queue:
                    node = queue.popleft()
                    new_component_1.append(node)
                    queue.extend(tree[node])
                new_component_2 = [node for node in component if node not in new_component_1]
                component = new_component_1
                return [component, new_component_2]
            else:
                # amikor egyik reszfa sem mely, mindegyik el az alapfa gyokerehez huzodik
                # addig vagunk le a levelekbol amig egy k meretu csoportot nem kapunk
                print("NEGYEDIK ESET")
                new_component_1 = []
                queue = deque(tree[u])
                size = 0
                while queue and size < k:
                    node = queue.popleft()
                    new_component_1.append(node)
                    size += subtree_sizes[node]
                    queue.extend(tree[node])
                new_component_2 = [node for node in component if node not in new_component_1]
                component = new_component_1
                return [component, new_component_2]
        else:
            # ebben az esetben ujra kellene definialji a fat s az eleket
            # ritka eset, kicsi k-nak fordul elo
            print("NINCS VAGAS")
            break

    return [component]


def run(data, n, k, adjacency_list, parents, distance_matrix):
    components = forest(n, k, adjacency_list, parents, distance_matrix)
    print("nr of componenets:", len(components))
    print("adjency", adjacency_list)
    components_list = list(components.values())

    final_groups = []
    for p, component in enumerate(components_list):
        print(p, ".component:", component)
        decomposed = decompose_component(list(component), k, adjacency_list)
        print(p, ". decomposed component:", decomposed)
        final_groups.extend(decomposed)

    for i, group in enumerate(final_groups):
        if len(group) > 2 * k - 1:
            centroid = np.mean(data[group], axis=0)
            distances = [(i, euclidean(data[i], centroid)) for i in group]
            distances.sort(key=lambda x: x[1], reverse=True)
            u = distances[0][0]
            nearest_neighbors = get_k_nearest_neighbors(u, k - 1, distance_matrix)
            new_group = [u] + list(nearest_neighbors)
            remaining_group = [x for x in group if x not in new_group]
            final_groups[i] = new_group
            final_groups.append(remaining_group)

    return final_groups


def main():
    pd.options.mode.chained_assignment = None  # default='warn'
    records = read_data_normalized()
    k = 31
    n = len(records)
    adjacency_list = defaultdict(list)
    parents = [-1] * n
    dm = distance_matrix(records, records)

    groups = run(records, n, k, adjacency_list, parents, dm)
    for group in groups:
        print("gr len:", len(group))

    cluster_assignment = np.zeros(len(records), dtype=int)

    # Assign cluster labels
    for cluster_id, record_indices in enumerate(groups):
        for record_index in record_indices:
            cluster_assignment[record_index] = cluster_id

    print("Cluster assignment array:", cluster_assignment)


def main2():
    data = np.array([
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 2.0],
        [8.0, 8.0],
        [8.5, 8.5],
        [9.0, 9.0],
        [1.0, 8.0],
        [1.5, 8.5],
        [2.0, 9.0],
        [8.0, 1.0]
    ])
    n = len(data)
    k = 3
    adjacency_list = defaultdict(list)
    parents = [-1] * n
    dm = distance_matrix(data, data)
    print(forest(n, k, adjacency_list, parents, dm))
    print("adjency", adjacency_list)



if __name__ == "__main__":
    main()

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean
from collections import defaultdict, deque
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_data_normalized():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
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


class MultivariateMicroaggregation:
    def __init__(self, data, k):
        self.data = np.array(data)
        self.k = k
        self.n = len(data)
        self.distance_matrix = distance_matrix(self.data, self.data)
        self.adjacency_list = defaultdict(list)
        self.parents = [-1] * self.n  # azt hiszem nem hasznalom sehol
        self.aggregated_result = []

    def _get_k_nearest_neighbors(self, u, k):
        dists = self.distance_matrix[u]
        nearest_neighbors = np.argsort(dists)
        return nearest_neighbors[1:k + 1]

    def _forest(self):
        # megszamozzuk mindegyik rekordot(csucst)
        # Az elejen mindegyik (csucs)rekord egy komponens
        components = {i: {i} for i in range(self.n)}
        vertices = set(range(self.n))
        components_ind = list(range(self.n))

        while any(len(comp) < self.k for comp in components.values()):
            for u in vertices:
                if len(components[u]) >= self.k:
                    continue
                nearest_neighbors = self._get_k_nearest_neighbors(u, self.k)
                for v in nearest_neighbors:
                    if v not in components[u]:
                        # a dict-ben szeretnenk ha ugyanaz a lista lenne mindegyik tagnal egy adott komponensben
                        components[v].update(components[u])
                        for w in components[u]:
                            components[w] = components[v]
                            components_ind[w] = components_ind[v]

                        self.adjacency_list[u].append(v)
                        self.parents[v] = u
                        break

        components_ind = list(set(components_ind))
        reduced_components = {i: components[j] for i, j in enumerate(components_ind)}
        return reduced_components

    def _decompose_component(self, component):
        k = self.k
        max_size = max(2 * k - 1, 3 * k - 5)

        while len(component) > max_size:
            # Azt a csucsot valasszuk amelyikbe nem megy el
            vertexes = missing_keys = [num for num in component if num not in self.adjacency_list]
            u = vertexes[0]

            # Step 3: Letrehozzuk a komponens-fat, majd a reszfak hosszat hatarozzuk meg
            tree = defaultdict(list)
            queue = deque([u])
            visited = set([u])

            while queue:
                node = queue.popleft()
                neighbours = [key for key, value_list in self.adjacency_list.items() if node in value_list]
                for neighbor in neighbours:
                    if neighbor in component and neighbor not in visited:
                        tree[node].append(neighbor)
                        queue.append(neighbor)
                        visited.add(neighbor)

            # print("tree",tree)
            subtree_sizes = {}

            def compute_subtree_sizes(node):
                size = 1
                for child in tree[node]:
                    size += compute_subtree_sizes(child)
                subtree_sizes[node] = size
                return size

            compute_subtree_sizes(u)
            del subtree_sizes[u]
            # print("subtree_sizes", subtree_sizes)

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

    # def _microaggregate(self, groups):
    #     for group in groups:
    #         centroid = np.mean(self.data[group], axis=0)
    #         for record in group:
    #             self.data[record] = centroid

    def _microaggregate(self, groups):
        for group in groups:
            centroid = np.mean(self.data[group], axis=0)
            self.aggregated_result.append(centroid)

    def run(self):
        components = self._forest()
        print("nr of componenets:", len(components))
        # print("components", components)
        print("adjency", self.adjacency_list)
        # print("parents", self.parents)
        components_list = list(components.values())
        # print("component list", components_list)

        final_groups = []
        for p, component in enumerate(components_list):
            print(p, ".component:", component)
            decomposed = self._decompose_component(list(component))
            print(p, ". decomposed component:", decomposed)
            final_groups.extend(decomposed)

        for i, group in enumerate(final_groups):
            if len(group) > 2 * self.k - 1:
                centroid = np.mean(self.data[group], axis=0)
                distances = [(i, euclidean(self.data[i], centroid)) for i in group]
                distances.sort(key=lambda x: x[1], reverse=True)
                u = distances[0][0]
                nearest_neighbors = self._get_k_nearest_neighbors(u, self.k - 1)
                new_group = [u] + list(nearest_neighbors)
                remaining_group = [x for x in group if x not in new_group]
                final_groups[i] = new_group
                final_groups.append(remaining_group)

        self._microaggregate(final_groups)
        return self.aggregated_result


def main():
    records = read_data_normalized()
    data = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [8.0, 8.0],
        [7.0, 5.0],
        [3.0, 3.5],
        [6.0, 7.0],
        [4.0, 4.5],
        [5.5, 6.5],
        [6.0, 11.0],
        [54.0, 4.5],
        [5.5, 5.5]
    ]
    k = 40

    mm = MultivariateMicroaggregation(records, k)
    result = mm.run()
    print("len of result:", len(result))
    print("aggregated result:", result)


if __name__ == "__main__":
    main()

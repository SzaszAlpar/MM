import numpy as np


def get_centroid(records, nn):
    record_number = len(records)
    centroid = np.zeros(nn)
    if record_number == 0:
        return centroid
    else:
        return np.average(records, axis=0)


def get_eucl(a, b):
    return np.linalg.norm(a - b)


def get_furthest(records, vect):
    furthest = None
    max_distance = 0.0
    furthest_index = 0

    for index, record in enumerate(records):
        dist = get_eucl(vect, record)
        if dist > max_distance:
            furthest = record
            max_distance = dist
            furthest_index = index

    return [furthest, furthest_index]


def get_closest_from_vector(records, vector):
    closest = None
    closest_index = 0
    min_distance = float('inf')

    for index, record in enumerate(records):
        dist = get_eucl(vector, record)
        if dist < min_distance:
            closest = record
            min_distance = dist
            closest_index = index

    return [closest, closest_index, min_distance]


def get_closest_from_group(formed_group, remaining_records):
    distance_list = []

    for index, gr_record in enumerate(remaining_records):
        distances = [get_eucl(vec, gr_record) for vec in formed_group]
        shortest_distance = min(distances)
        distance_list.append(shortest_distance)

    distance_list = np.array(distance_list)
    closest_index = np.argmin(distance_list, axis=0)
    min_distance = distance_list[closest_index]
    closest = remaining_records[closest_index]

    return [closest, closest_index, min_distance]


def get_closest_k(records, vec, k):
    group_distances = np.array([get_eucl(rec, vec) for rec in records])
    group_indexes = np.array(range(0, len(records)))
    argsort = group_distances.argsort()
    records_cpy = records.copy()
    records_cpy = records_cpy[argsort]
    group_indexes = group_indexes[argsort]
    return [records_cpy[0:k], group_indexes[0:k]]


def append_to_closest_group(records, groups, record_indices, indices):
    nn = records.shape[1]
    centroids = [get_centroid(group, nn) for group in groups]
    for ind, record in enumerate(records):
        distances = [get_eucl(centroid, record) for centroid in centroids]
        closest_index = np.argmin(distances)
        groups[closest_index] = np.vstack([groups[closest_index], record])
        indices[closest_index] = np.append(indices[closest_index], record_indices[ind])
    return groups


def should_add_record(d_in, d_out, gamma=0.2):
    return d_in <= d_out * gamma


def V_MDAV(records, k):
    nn = len(records[0])
    RR = len(records)
    print("Starting with {} records".format(RR))
    groups = []
    indices = []
    record_indices = np.arange(RR)
    centroid = get_centroid(records, nn)

    while RR > k - 1:
        [r, r_ind] = get_furthest(records, centroid)

        [gr, gr_ind] = get_closest_k(records, r, k)
        groups.append(gr.copy())
        indices.append(record_indices[gr_ind].copy())
        records = np.delete(records, gr_ind, 0)
        record_indices = np.delete(record_indices, gr_ind, 0)
        if len(records) == 0:
            break

        # extend this group if there are very close records
        e_in, e_in_index, d_in = get_closest_from_group(gr, records)
        e_out, e_out_index, d_out = get_closest_from_vector(records, e_in)
        gr_extension = []
        gr_extension_indexes = []
        # while there are records that are closer to this group than to the unassigned records
        while should_add_record(d_in, d_out) and len(gr_extension) < k - 1:
            gr_extension.append(e_in.copy())
            gr_extension_indexes.append(record_indices[e_in_index].copy())
            records = np.delete(records, e_in_index, 0)
            record_indices = np.delete(record_indices, e_in_index, 0)
            if len(records) == 0:
                break

            e_in, e_in_index, d_in = get_closest_from_group(gr, records)
            e_out, e_out_index, d_out = get_closest_from_vector(records, e_in)

        if len(gr_extension) > 0:
            idx = len(groups) - 1
            groups[idx] = np.vstack([groups[idx], np.array(gr_extension)])
            indices[idx] = np.append(indices[idx], np.array(gr_extension_indexes), axis=0)

        RR = len(records)
        print("Remaining records, RR = ", RR)

    if RR > k:
        groups.append(records.copy())
    else:
        groups = append_to_closest_group(records, groups, record_indices, indices)

    return groups, indices

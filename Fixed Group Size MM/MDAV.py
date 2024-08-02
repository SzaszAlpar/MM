import numpy as np
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


def get_centroid(records, nn):
    record_number = len(records)
    centroid = np.zeros(nn)
    if record_number == 0:
        return centroid
    else:
        return np.average(records, axis=0)


def get_eucl(a, b):
    return np.linalg.norm(a - b)


def get_furthest(records, vector, nn):
    furthest = np.zeros(nn)
    max_distance = 0.0
    furthest_index = 0
    index = 0

    for record in records:
        dist = get_eucl(vector, record)
        if dist > max_distance:
            furthest = record
            max_distance = dist
            furthest_index = index
        index += 1

    return [furthest, furthest_index]


def get_closest_k(records, vec, k):
    group_distances = np.array([get_eucl(rec, vec) for rec in records])
    group_indexes = np.array(range(0, len(records)))
    argsort = group_distances.argsort()
    records = records[argsort]
    group_indexes = group_indexes[argsort]
    return [records[0:k], group_indexes[0:k]]


def aggregate(groups):
    result = []
    nn = len(groups[0][0])
    for group in groups:
        result.append(get_centroid(group, nn))
    return result


def append_to_closest_group(records, groups):
    nn = records.shape[1]
    centroids = [get_centroid(group, nn) for group in groups]
    for record in records:
        distances = [get_eucl(centroid, record) for centroid in centroids]
        closest_index = np.argmin(distances)
        groups[closest_index] = np.append(groups[closest_index], record)
    return groups


def MDAV(records, k):
    nn = len(records[0])
    RR = len(records)
    print("Starting with {} records".format(RR))
    groups = []
    while RR > 2 * k:
        centroid = get_centroid(records, nn)

        [r, r_ind] = get_furthest(records, centroid, nn)
        [gr, gr_ind] = get_closest_k(records, r, k)
        print("gr_ind1", gr_ind)
        groups.append(gr.copy())
        records = np.delete(records, gr_ind, 0)

        [s, s_ind] = get_furthest(records, r, nn)
        [gr2, gr_ind2] = get_closest_k(records, s, k)
        print("gr_ind2", gr_ind2)
        groups.append(gr2.copy())
        records = np.delete(records, gr_ind2, 0)

        RR = len(records)
        print("Remaining records, RR = ", RR)
    print("Ungrouped records, RR = ", RR)
    if RR > k:
        groups.append(records.copy())
    else:
        groups = append_to_closest_group(records, groups)

    return groups


def main():
    records = read_data_normalized()
    k = 40
    groups = MDAV(records, k)
    print("groups len:", len(groups))
    print("aggregated result:", aggregate(groups))
    print("type", type(groups[0]))


if __name__ == "__main__":
    main()

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def update_centroid(cluster, numeric_indices, categorical_indices):
    centroid = [0] * (len(numeric_indices) + len(categorical_indices))

    for i in numeric_indices:
        sum_dim = sum(point[i] for point in cluster)
        mean_dim = sum_dim / len(cluster)
        centroid[i] = mean_dim

    for i in categorical_indices:
        categories = [point[i] for point in cluster]
        most_common = max(set(categories), key=categories.count)
        centroid[i] = most_common

    return centroid


def k_means_mixed(data, k, numeric_indices, categorical_indices, max_iter=100):
    random.seed(999)
    centroids = [random.choice(data) for _ in range(k)]
    cluster_labels = [-1 for _ in range(len(data))]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for idx, point in enumerate(data):
            distances = [mixed_distance(point, centroid, numeric_indices, categorical_indices) for centroid in centroids]
            min_distance_index = distances.index(min(distances))
            clusters[min_distance_index].append(point)
            cluster_labels[idx] = min_distance_index
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroid = choose_farthest_point(data, centroids, numeric_indices, categorical_indices)
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(update_centroid(cluster, numeric_indices, categorical_indices))
        centroids = new_centroids

    return centroids, cluster_labels


def choose_farthest_point(data, centroids, numeric_indices, categorical_indices):
    max_distance = 0
    farthest_point = None
    for point in data:
        distance = sum([mixed_distance(point, centroid, numeric_indices, categorical_indices) for centroid in centroids])
        if distance > max_distance:
            max_distance = distance
            farthest_point = point
    return farthest_point


def mixed_distance(point1, point2, numeric_indices, categorical_indices):
    num_distance = sum((point1[i] - point2[i]) ** 2 for i in numeric_indices) ** 0.5
    cat_distance = sum(point1[i] != point2[i] for i in categorical_indices)
    return num_distance + cat_distance


def calculate_center_and_radius(gb, numeric_indices, categorical_indices):
    center = [0] * (len(numeric_indices) + len(categorical_indices))
    n_gb = len(gb)
    for i in numeric_indices:
        sum_dim = sum(point[i] for point in gb)
        center[i] = sum_dim / n_gb

    for i in categorical_indices:
        categories = [point[i] for point in gb]
        most_common = max(set(categories), key=categories.count)
        center[i] = most_common
    
    radius = 0
    for s in gb: 
        t = sum((s[i] - center[i]) ** 2 for i in numeric_indices) ** 0.5
        t += sum(s[i] != center[i] for i in categorical_indices)
        radius += t
    
    return center, radius / n_gb


def splits(gb_list, num, k, flag):
    gb_list_new = []
    for gb in gb_list:
        p = gb.shape[0]
        if p < num:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb, k, flag))
    return gb_list_new


def splits_ball(gb, k, flag):
    ball_list = []
    len_no_label = np.unique(gb, axis=0)
    if len_no_label.shape[0] < k:
        k = len_no_label.shape[0]

    t_n, t_m = gb.shape
    attrs = np.arange(t_m)
    num_idxs = attrs[flag]
    cat_idxs = attrs[~flag]
    _, label = k_means_mixed(gb, k, num_idxs, cat_idxs)
    label = np.array(label)
    for single_label in range(0, k):
        ball_list.append(gb[label == single_label, :])
    return ball_list
        
    
def assign_points_to_closest_gb(data, gb_centers, num_idxs, cat_idxs):
    assigned_gb_indices = np.zeros(data.shape[0])
    for idx, sample in enumerate(data):
        t_idx = -1
        dist = 99999
        for jdx, center in enumerate(gb_centers):
            cur_dist = mixed_distance(sample, center, num_idxs, cat_idxs)
            if cur_dist < dist:
                dist = cur_dist
                t_idx = jdx
        assigned_gb_indices[idx] = t_idx
    
    return assigned_gb_indices.astype('int')


def fuzzy_similarity(t_data, sigma, k=2):
    t_n, t_m = t_data.shape
    t_flag = (t_data <= 1).all(axis=0) & (t_data.max(axis=0) != t_data.min(axis=0))
    
    attrs = np.arange(t_m)
    num_idxs = attrs[t_flag]
    cat_idxs = attrs[~t_flag]
    
    gb_list = [t_data]
    num = np.ceil(t_n ** 0.5)
    while True:
        ball_number_1 = len(gb_list)
        gb_list = splits(gb_list, num, k, t_flag)
        ball_number_2 = len(gb_list)
        if ball_number_1 == ball_number_2:
            break
    gb_center = np.zeros((len(gb_list), t_m))
    
    for idx, gb in enumerate(gb_list):
        gb_center[idx], _ = calculate_center_and_radius(gb, num_idxs, cat_idxs)
    
    point_to_gb = assign_points_to_closest_gb(t_data, gb_center, num_idxs, cat_idxs)
    
    n_gb = len(gb_center)
    dist = np.zeros((n_gb, n_gb))
    for i in range(n_gb):
        for j in range(n_gb):
            dist[i, j] = mixed_distance(gb_center[i], gb_center[j], num_idxs, cat_idxs)
    fs = 1 - dist / t_m
    fs[fs < sigma] = 0
    return fs, point_to_gb
    

def HGBAD(data, sigma=1):
    n, m = data.shape
    E = np.zeros(m)
    OD = np.zeros((n, m))
    point_fs = np.zeros((n, n))

    for i in range(m):
        gb_fs, point_to_gb = fuzzy_similarity(data[:,[i]], sigma, 2)
        for s in range(n):
            for t in range(s + 1):
                point_fs[s, t] = gb_fs[point_to_gb[s], point_to_gb[t]]
                point_fs[t, s] = point_fs[s, t]
        
        E[i] = -np.sum((1 / n) * np.log2(np.sum(point_fs, 1) / n))
        for j in range(n):
            OD[j,i] = 1 - point_fs[j].mean()

    W = E / np.sum(E)
    OF = np.mean(OD * np.cbrt(W), axis=1)
    return OF


if __name__ == '__main__':
    data = pd.read_csv("./Example.csv").values
    sigma = 0.6
    out_factors = HGBAD(data, sigma)
    print(out_factors)

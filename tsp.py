import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

def get_center_from_bbox(bbox):
    #TODO: Implement for real data
    pass


def create_dummy_points(grid_size=10, num_points=10):
    prng = RandomState(1234567890)
    x = prng.randint(0, grid_size, num_points)
    y = prng.randint(0, grid_size, num_points)


    _, ax = plt.subplots()
    ax.scatter(x, y)
    for i in range(num_points):
        ax.annotate(i, (x[i], y[i]))

    return x,y

def get_dist_from_points(x1, y1, x2, y2):
    x_ = np.square(x1 - x2)
    y_ = np.square(y1 - y2)

    return float(np.sqrt(x_ + y_))

def convert_coords_to_dist_matrix(x, y):
    #TODO: Can I get away without creating this distance matrix?
    d = []
    for i in range(len(x)):
        tmp = []
        for j in range(len(x)):
            if i == j:
                tmp.append(0)
            else:
                tmp.append(get_dist_from_points(x[i], y[i], x[j], y[j]))
        d.append(tmp)

    # plt.figure(2)
    # data_array = np.array(d)
    # plt.imshow(data_array, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.show()
    return d


def basic_tsp(d):
    # Time complexity O(n!); Optimal

    pass

def tsp_nearest_neighbor(distances):
    # Time complexity O(n^2); Greedy
    # https://www.w3schools.com/dsa/dsa_ref_traveling_salesman.php

    n = len(distances)
    visited = [False] * n
    route = [0]
    visited[0] = True
    total_distance = 0

    for _ in range(1, n):
        last = route[-1]
        nearest = None
        min_dist = float('inf')
        for i in range(n):
            if not visited[i] and distances[last][i] < min_dist:
                min_dist = distances[last][i]
                nearest = i
        route.append(nearest)
        visited[nearest] = True
        total_distance += min_dist

    total_distance += distances[route[-1]][0]
    route.append(0)
    return route, total_distance


def tsp_bucket_sort():
    pass

x, y = create_dummy_points(10, 5)
d = convert_coords_to_dist_matrix(x, y)
order_nearest_neigh = tsp_nearest_neighbor(d)
print(order_nearest_neigh)

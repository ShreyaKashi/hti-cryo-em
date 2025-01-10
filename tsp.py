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
    # plt.show()
    return d


def basic_tsp(d):
    # Time complexity O(n!); Optimal

    pass

def tsp_nearest_neighbor(d):
    # Time complexity O(n^2); Greedy
    pass

def tsp_bucket_sort():
    pass

x, y = create_dummy_points(10, 5)
d = convert_coords_to_dist_matrix(x, y)
order_nearest_neigh = tsp_nearest_neighbor(d)

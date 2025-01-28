from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from qthist2d import qthist, qtcount
from matplotlib.path import Path
import time

def get_center_from_bbox(bbox):
    #TODO: Implement for real data
    # The attributes are "boxes", a list of bounding box coordinates (x_min, y_min, x_max, y_max) and 
    # "scores", a list of corresponding score

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


def basic_tsp(cost):
    # Time complexity O(n!); Optimal

    numNodes = len(cost)
    nodes = list(range(1, numNodes))

    minCost = float('inf')

    for perm in permutations(nodes):
        currCost = 0
        currNode = 0

        for node in perm:
            currCost += cost[currNode][node]
            currNode = node

        currCost += cost[currNode][0]

        minCost = min(minCost, currCost)

    return minCost

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


def tsp_bucket_sort(x,y):

    graph = {}

    def create_graph():
        pass

    num, xmin, xmax, ymin, ymax = qthist(x,y, N=5, thresh=4)
    pts = np.column_stack((x,y))
    fig = plt.figure()

    ax = fig.add_subplot(111)

    plt.scatter(x,y, alpha=0.5)
    

    for k in range(len(num)):
        ax.add_patch(plt.Rectangle((xmin[k], ymin[k]), xmax[k]-xmin[k], ymax[k]-ymin[k], 
                                fc ='none', ec='k', alpha=0.5))
        
        ll = [xmin[k], ymin[k]]
        ur = [xmax[k], ymax[k]]
        inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
        inbox = pts[inidx]
        print(inbox)
        
    


x, y = create_dummy_points(10, 5)
d = convert_coords_to_dist_matrix(x, y)

start = time.time()
order_original_tsp = basic_tsp(d)
end = time.time()
print("Original TSP: ", end - start)

start = time.time()
order_nearest_neigh = tsp_nearest_neighbor(d)
end = time.time()
print("Nearest Neighbor: ", end - start)

print("Order: ", order_original_tsp, order_nearest_neigh)
# tsp_bucket_sort(x,y)
# plt.show()

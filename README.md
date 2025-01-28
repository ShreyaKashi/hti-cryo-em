# Optimizing microscope control

Cryo-EM is a technique that involves studying frozen samples deposited within a grid of squares and holes through a microscope. Not all squares and holes have good quality samples which would help in diagnosis. Some, based on how they were frozen, have high noise and are not very useful to study. Since using a cryoelectron microscope is expensive, we wish to optimize microscope control to efficiently traverse over all the good quality squares/holes during data collection. 

One way to look at this problem is by viewing the squares/holes as a graph and to model the problem as finding an efficient graph traversal algorithm. This means we wish to find a Hamiltonian path with the least weight in an undirected graph. A Hamiltonian path in a graph is a path that visits each vertex exactly once and the weight is defined as the overall distance travelled.

Algorithm summary:

- Create a fully connect graph
- Find all Hamiltoinian paths
- Select one with least total distance

The issue with this algorithm is that it is computationally intractable. Exploring all possible permutations in a fully connected graph requires O(n!) time.

This problem is often called the Travelling Salesman problem and has different approximations that make it computationally more efficient but less accurate.

One such approximation is the Nearest Neighbor approximation. 

Algorithm summary:
- Create a fully connect graph
- For each node find the nearest node and add it to the path

This algorithm runs in O(n^2) time.

Another approximation is to prune the fully-connected graph. Here we explore using adaptive binning to divide the graph into clusters. Each cluster is fully connected within itself and fully connected to every nodeof its immediate neighboring cluster. 

Algorithm summary:
- Compute bins based on x and y coordinates
- Store bin_id, x, y
- For all x, y in same bin_id, computer distance using convert_coords_to_dist_matrix
- Use original TSP to compute time



Find shortest Hamiltonion cycle [1]
"Given a directed graph G = (V, E), we say that a cycle C in G is a Hamiltonian cycle if it visits each
vertex exactly once."

[1] https://edisciplinas.usp.br/pluginfile.php/7933913/course/section/6549987/Algorithm%20Design.pdf
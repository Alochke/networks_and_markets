# Skeleton file for HW2 question 10
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random] 
# and UndirectGraph from hw2_p9
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from hw2_p9 import UndirectedGraph

# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class WeightedDirectedGraph(UndirectedGraph):
    def set_edge(self, origin_node, destination_node, weight=1):
        ''' Modifies the weight for the specified directed edge, from origin to destination node,
            with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from 
            the graph. If edge previously wasn't in the graph, adds a new edge with specified weight.'''
        self.adj_matrix[origin_node][destination_node] = weight
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise.'''
        return self.adj_matrix[origin_node][destination_node]
    
def shortest_path(G: WeightedDirectedGraph, s: int, t: int):
    """
    Finds the shortest path from node s to node t using a BFS algorithm.
    Args:
        G (WeightedDirectedGraph): The graph.
        s (int): Start node.
        s (int): End node.
    Returns:
        list[int] or none: The shortest path from s to t, where if the path is v_1->...->v_n,
            we'll return the list [v_1,...,v_n]
            or None if no path exists.
    """
    if s == t:  # Check if start and end nodes are the same
        return []

    # Initialize a queue for BFS
    queue = [(s, [s])]  # Each element is a tuple (current_node, path)
    visited = set([s])  # Keep track of visited nodes to prevent revisiting

    while queue:
        current_node, current_path = queue.pop(0)

        # Iterate through each neighbor of the current node
        for neighbor in G.edges_from(current_node):
            if neighbor == t:
                return current_path + [t]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_path + [neighbor]))

    return None # Return None if no path is found

# === Problem 10(a) ===
def max_flow(G: WeightedDirectedGraph, s: int, t: int):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
       Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
       for G, represented by another WeightedDirectedGraph where edge weights represent
       the final allocated flow along that edge.'''
    extra = 0
    map = {}
    def execute(size: int, func, triangle: bool):
        for i in range(size):
            for j in range(i + 1 if triangle else 0, size):
                func(i, j)
    def func1(i, j):
        nonlocal extra
        if G.get_edge(i, j) and G.get_edge(j, i) and i != j:
            map[extra] = [i, j]
            extra += 1
    execute(G.num_nodes, func1, True)
    G2 = WeightedDirectedGraph(G.num_nodes + extra)
    execute(G.num_nodes, lambda i, j: G2.set_edge(i, j, G.get_edge(i, j)) if i != j else None, False)
    for key, val in map.items():
        G2.set_edge(val[0], key + G.num_nodes, G.get_edge(val[0], val[1]))
        G2.set_edge(key + G.num_nodes, val[1], G.get_edge(val[0], val[1]))
        G2.set_edge(val[0], val[1], 0)

    G2_flow_graph = WeightedDirectedGraph(G2.num_nodes)
    def create_residual(G: WeightedDirectedGraph, flow_graph: WeightedDirectedGraph):
        ret = WeightedDirectedGraph(G.num_nodes)
        execute(G.num_nodes, lambda i, j: ret.set_edge(i, j, G.get_edge(i, j) - flow_graph.get_edge(i, j) if G.check_edge(i, j) else flow_graph.get_edge(j, i)), False)
        return ret
    while True:
        G_f = create_residual(G2, G2_flow_graph)
        path = shortest_path(G_f, s, t)
        if not path:
            break
        bottleneck = min([G_f.get_edge(path[i], path[i + 1]) for i in range(len(path) - 1)])
        for i in range(len(path) - 1):
            if G.get_edge(path[i], path[i + 1]):
                G2_flow_graph.adj_matrix[path[i], path[i + 1]] += bottleneck
            else:
                G2_flow_graph.adj_matrix[path[i + 1], path[i]] -= bottleneck

    G_flow_graph = WeightedDirectedGraph(G.num_nodes)
    execute(G.num_nodes, lambda i, j: G_flow_graph.set_edge(i, j, G2_flow_graph.get_edge(i, j)), False)

    for key, val in map.items():
        G_flow_graph.set_edge(val[0], val[1], G2_flow_graph.get_edge(val[0], G.num_nodes + key))

    return (sum(G_flow_graph.adj_matrix[s]) - sum(G_flow_graph.adj_matrix[:,s]), G_flow_graph)

# === Problem 10(c) ===
def max_matching(n, m, C):
    '''Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0. 
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched.'''
    # TODO: Implement this method
    pass

# === Problem 10(d) ===
def random_driver_rider_bipartite_graph(n, p):
    '''Returns an n x n constraints array C as defined for max_matching, representing a bipartite
       graph with 2n nodes, where each vertex in the left half is connected to any given vertex in the 
       right half with probability p.'''
    # TODO: Implement this method
    pass

def main():
    # TODO: Put your analysis and plotting code here for 10(d)
    pass

if __name__ == "__main__":
    main()

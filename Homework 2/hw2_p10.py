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
class WeightedDirectedGraph:
    def __init__(self,number_of_nodes):
        '''Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1.'''
        # TODO: Implement this method
        pass
    
    def set_edge(self, origin_node, destination_node, weight=1):
        ''' Modifies the weight for the specified directed edge, from origin to destination node,
            with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from 
            the graph. If edge previously wasn't in the graph, adds a new edge with specified weight.'''
        # TODO: Implement this method
        pass
    
    def edges_from(self, origin_node):
        ''' This method shold return a list of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph (i.e. with weight > 0).'''
        # TODO: Implement this method
        pass
    
    def get_edge(self, origin_node, destination_node):
        ''' This method should return the weight (an integer > 0) 
            if there is an edge between origin_node and 
            destination_node, and 0 otherwise.'''
        # TODO: Implement this method
        pass
    
    def number_of_nodes(self):
        ''' This method should return the number of nodes in the graph'''
        # TODO: Implement this method
        pass

# === Problem 10(a) ===
def max_flow(G, s, t):
    '''Given a WeightedDirectedGraph G, a source node s, a destination node t,
       compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
       Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
       for G, represented by another WeightedDirectedGraph where edge weights represent
       the final allocated flow along that edge.'''
    # TODO: Implement this method
    pass

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

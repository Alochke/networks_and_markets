# Skeleton file for HW2 question 9
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt

# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class UndirectedGraph:
    def __init__(self, number_of_nodes):
        """
        Initialize the graph with the specified number of nodes. Each node is identified by an integer index.
        Uses an adjacency matrix for storage where each cell (i, j) indicates the presence of an edge between nodes i and j.
        Args:
            number_of_nodes (int): The total number of nodes in the graph. Nodes are numbered from 0 to number_of_nodes - 1.
        """
        self.num_nodes = number_of_nodes
        self.adj_matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=int)
    
    def add_edge(self, nodeA, nodeB):
        """
        Add an undirected edge to the graph between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        """
        self.adj_matrix[nodeA][nodeB] = 1
        self.adj_matrix[nodeB][nodeA] = 1
    
    def edges_from(self, nodeA):
        """
        Return a list of all nodes connected to nodeA by an edge.
        Args:
            nodeA (int): Index of the node to retrieve edges from.
        Returns:
            list[int]: List of nodes that have an edge with nodeA.
        """
        return list(np.where(self.adj_matrix[nodeA] == 1)[0])
    
    def check_edge(self, nodeA, nodeB):
        """
        Check if there is an edge between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        Returns:
            bool: True if there is an edge, False otherwise.
        """
        return self.adj_matrix[nodeA][nodeB] == 1
    
    def number_of_nodes(self):
        """
        Return the number of nodes in the graph.
        Returns:
            int: The number of nodes in the graph.
        """
        return self.num_nodes
    
    def print_graph(self):
        print("Adjacency Matrix:")
        print(self.adj_matrix)

def create_fb_graph(filename = "facebook_combined.txt"):
    ''' This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes.'''    
    # TODO: Implement this method
    pass


# === Problem 9(a) ===

def contagion_brd(G, S, t):
    '''Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
       and a float threshold t, perform BRD as follows:
       - Permanently infect the nodes in S with X
       - Infect the rest of the nodes with Y
       - Run BRD on the set of nodes not in S
       Return a list of all nodes infected with X after BRD converges.'''
    # TODO: Implement this method
    pass

def q_completecascade_graph_fig4_1_left():
    '''Return a float t s.t. the left graph in Figure 4.1 cascades completely.'''
    # TODO: Implement this method 
    pass

def q_incompletecascade_graph_fig4_1_left():
    '''Return a float t s.t. the left graph in Figure 4.1 does not cascade completely.'''
    # TODO: Implement this method 
    pass

def q_completecascade_graph_fig4_1_right():
    '''Return a float t s.t. the right graph in Figure 4.1 cascades completely.'''
    # TODO: Implement this method 
    pass

def q_incompletecascade_graph_fig4_1_right():
    '''Return a float t s.t. the right graph in Figure 4.1 does not cascade completely.'''
    # TODO: Implement this method 
    pass

def main():
    # === Problem 9(b) === #
    # TODO: Put analysis code here
    # === Problem 9(c) === #
    # TODO: Put analysis code here
    # === OPTIONAL: Bonus Question 2 === #
    # TODO: Put analysis code here
    pass

# === OPTIONAL: Bonus Question 2 === #
def min_early_adopters(G, q):
    '''Given an undirected graph G, and float threshold t, approximate the 
       smallest number of early adopters that will call a complete cascade.
       Return an integer between [0, G.number_of_nodes()]'''
    pass

if __name__ == "__main__":
    main()

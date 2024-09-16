import numpy as np
import os
import numpy as np
from typing import Union

##########################################################################################
class DirectedGraph:
    def __init__(self, number_of_nodes, wheighted: bool):
        self.adj_matrix: np.ndarray = np.zeros((number_of_nodes, number_of_nodes), dtype=np.int32 if wheighted else np.bool)
    
    def add_edge(self, origin_node, destination_node, weight = True):
        '''Adds an edge from origin_node to destination_node.'''
        self.add_edge[origin_node][destination_node] = weight
    
    def edges_from(self, origin_node):
        ''' Returns a one-dimentional np array of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph.'''
        if self.adj_matrix.dtype == np.int32:
            return np.where(self.adj_matrix[origin_node] > 0)[0]
        else:
            return np.where(self.adj_matrix[origin_node] == True)[0]
    
    def get_edge(self, origin_node, destination_node):
        ''' Returns the value on the edge forom origin_node to destination_node, will be false or 0 if there's no such edge.'''
        return self.adj_matrix[origin_node][destination_node]
    
    def number_of_nodes(self):
        ''' Returns the number of nodes in the graph.'''
        return self.adj_matrix.shape[0]
    
    def transpose(self):
        ''' Returns a new DirectedGraph that is the transpose of the current graph.'''
        return self.adj_matrix.T
    




##########################################################################################

def scaled_page_rank(G, num_iter, eps=1/7.0):
    '''
    This method, given a DirectedGraph G, runs the epsilon-scaled 
    page-rank algorithm for num-iter iterations, for parameter eps,
    and returns a Dictionary where the keys are the set of 
    nodes [0,...,G.number_of_nodes() - 1], each associated with a value
    equal to the score of output by the eps-scaled pagerank algorithm.

    In the case of num_iter=0, all nodes should 
    have weight 1/G.number_of_nodes()
    '''

    n = G.number_of_nodes()
    scores = np.array([1/n for i in range(n)])

    if num_iter == 0:
        return scores

    # Create a transposed graph to simplify the calculation of incoming edges
    transposed_G = G.transpose()

    # Compute out-degrees for each node
    out_degrees = {node: len(G.edges_from(node)) for node in range(n)}

    # Iterative computation of PageRank scores using transposed graph
    for _ in range(num_iter):
        new_scores = {}
        for v in range(n):
            incoming_score = sum(scores[v0] / out_degrees[v0] for v0 in transposed_G.edges_from(v))
            new_scores[v] = eps/n + (1 - eps) * incoming_score
        
        # Normalize scores to ensure they sum to 1
        total_score = sum(new_scores.values())
        scores = {node: score / total_score for node, score in new_scores.items()}

    return scores

# === Problem 6. ===

def graph_15_1_left():
    ''' This method constructs and returns a DirectedGraph encoding the left example in fig 15.1
    Use the following indexes mapped to names for clarity: A:0, B:1, C:2, Z:3 '''

    # Create a new DirectedGraph with 4 nodes
    graph = DirectedGraph(4)

    # Define node labels as variables for clarity
    A = 0
    B = 1
    C = 2
    Z = 3

    # Add edges using the variables to make it clear
    graph.add_edge(A, B)  # A -> B
    graph.add_edge(B, C)  # B -> C
    graph.add_edge(C, A)  # C -> A
    graph.add_edge(A, Z)  # A -> Z
    graph.add_edge(Z, Z)  # Z -> Z (self-loop)

    return graph

def graph_15_1_right():
    ''' This method constructs and returns a DirectedGraph encoding the right example in fig 15.1
    Use the following indexes mapped to names for clarity: A:0, B:1, C:2, Z1:3, Z2:4 '''

    # Create a new DirectedGraph with 5 nodes
    graph = DirectedGraph(5)

    # Define node labels as variables for clarity
    A = 0
    B = 1
    C = 2
    Z1 = 3
    Z2 = 4

    # Add edges using the variables to make it clear
    graph.add_edge(A, B)  # A -> B
    graph.add_edge(B, C)  # B -> C
    graph.add_edge(C, A)  # C -> A
    graph.add_edge(A, Z1) # A -> Z1
    graph.add_edge(A, Z2) # A -> Z2
    graph.add_edge(Z1, Z2) # Z1 -> Z2
    graph.add_edge(Z2, Z1) # Z2 -> Z1

    return graph

def graph_15_2():
    ''' This method constructs and returns a DirectedGraph encoding example 15.2
        Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5'''

    # Create a new DirectedGraph with 6 nodes
    graph = DirectedGraph(6)

    # Dictionary to map node labels to indices for clarity
    node_map = {
        'A': 0,
        'B': 1,
        'C': 2,
        "A'": 3,
        "B'": 4,
        "C'": 5
    }

    # Add edges using the dictionary keys
    graph.add_edge(node_map['A'], node_map['B'])  # A -> B
    graph.add_edge(node_map['B'], node_map['C'])  # B -> C
    graph.add_edge(node_map['C'], node_map['A'])  # C -> A

    graph.add_edge(node_map["A'"], node_map["B'"])  # A' -> B'
    graph.add_edge(node_map["B'"], node_map["C'"])  # B' -> C'
    graph.add_edge(node_map["C'"], node_map["A'"])  # C' -> A'

    return graph

def extra_graph_1():
    ''' Constructs and returns a DirectedGraph where each node points to the next,
        forming a cyclic structure. '''

    # Create a DirectedGraph with 10 nodes
    graph = DirectedGraph(10)

    # Add edges to form a cycle
    for node in range(10):
        next_node = (node + 1) % 10  # Connect each node to the next, wrap around to the first
        graph.add_edge(node, next_node)

    return graph

def extra_graph_2():
    ''' Constructs and returns a DirectedGraph with a more complex structure,
        including self-loops and parallel paths. '''

    # Create a DirectedGraph with 10 nodes
    graph = DirectedGraph(10)

    # Adding various edges, including self-loops and multiple edges from a single node
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),  # Creating a simple cycle
        (2,0), (3 ,0), (4,0), #more edges that connected to 0
        (0, 0), (1, 1), (5, 5)  # Self-loops
    ]
    for u, v in edges:
        graph.add_edge(u, v)

    return graph

# === Problem 8. ===
def facebook_graph(filename="facebook_combined.txt"):
    ''' This method returns a DIRECTED version of the Facebook graph as an instance of the DirectedGraph class.
        If u and v are friends, there should be an edge between u and v and an edge between v and u.'''

    # Start by reading the file to determine the maximum node index for graph initialization
    with open(filename, 'r') as file:
        nodes = set()
        for line in file:
            u, v = map(int, line.split())
            nodes.update([u, v])
    
    # Create a DirectedGraph with the appropriate number of nodes
    graph = DirectedGraph(max(nodes) + 1)  # +1 because nodes start at 0

    # Re-open the file to add edges
    with open(filename, 'r') as file:
        for line in file:
            u, v = map(int, line.split())
            graph.add_edge(u, v)
            graph.add_edge(v, u)  # Ensure bidirectional connection

    return graph

def visualize_pagerank_results(pagerank_results):
    # Create the directory for storing results if it doesn't exist
    results_dir = "pagerank_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Loop through each graph's PageRank results
    for graph_name, scores in pagerank_results.items():
        # Prepare data for plotting
        nodes = list(scores.keys())
        values = [scores[node] * 100 for node in nodes]  # Convert to percentages

        # Create a bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(nodes, values, color='skyblue')
        
        # Set x-axis labels to correspond directly to node identifiers
        plt.xticks(nodes, [f'Node {node}' for node in nodes])  # Adjust tick labels
        
        plt.xlabel('Node')
        plt.ylabel('PageRank (%)')
        plt.title(f'PageRank for {graph_name}')
        plt.ylim(0, max(values) + 5)  # Ensure there is some space above the highest bar

        # Save the plot to the designated folder
        plt.savefig(f"{results_dir}/{graph_name.replace(' ', '_')}_PageRank.png")
        plt.close()  # Close the figure context to free up memory

def question_8_section_b(iterations = 20, save_visualization = False):
    # List of graphs to process
    graphs = {
        "Graph 15.1 Left": graph_15_1_left(),
        "Graph 15.1 Right": graph_15_1_right(),
        "Graph 15.2": graph_15_2(),
        "Extra Graph 1": extra_graph_1(),
        "Extra Graph 2": extra_graph_2(),
    }

    # Initialize PageRank results
    pagerank_results = {}

    # Calculate PageRank for each graph 
    for name, graph in graphs.items():
        print(f"\nCalculating PageRank for {name}:")
        pagerank_scores = scaled_page_rank(graph, num_iter=iterations)  # Using 20 iterations as a common basis
        pagerank_results[name] = pagerank_scores

        # Print results for each node in the graph
        for node in sorted(pagerank_scores):
            print(f"Node {node}: {pagerank_scores[node]:.4f}")
        
        if save_visualization:
            visualize_pagerank_results(pagerank_results)

    return pagerank_results

# If this function is to be part of a larger script, it should ideally be called within a main block:

    
def main():
    # TODO: Put your analysis and plotting code here for 8(b)
    question_8_section_b(save_visualization=True)

    

if __name__ == "__main__":
    main()
import numpy as np
from typing import Union
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

TITLE_DS_PATH = "soc-redditHyperlinks-body.tsv"
BODY_DS_PATH = "soc-redditHyperlinks-title.tsv"
SUBREDDIT_INDEXING_PATH = "subbreddit_indexing.csv"
SOURCE_SUBREDDIT = 0
TARGET_SUBREDDIT = 1
NUM_ROWS = 0

##########################################################################################
class DirectedGraph:
    def __init__(self, number_of_nodes, weighted : bool):
        self.adj_matrix: np.ndarray = np.zeros((number_of_nodes, number_of_nodes), dtype=np.int16 if weighted  else np.bool)
    
    def add_edge(self, origin_node, destination_node, weight = True):
        '''Adds an edge from origin_node to destination_node.'''
        self.adj_matrix[origin_node][destination_node] = weight
    
    def edges_from(self, origin_node):
        ''' Returns a one-dimentional np array of all the nodes destination_node such that there is
            a directed edge (origin_node, destination_node) in the graph.'''
        return np.where(self.adj_matrix[origin_node] > 0)[0]
    
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

def page_rank_plus(G: DirectedGraph, num_iter, eps=1/7.0):
    """
    Run the epsilon-scaled improved PageRank algorithm on a directed graph for a specified number of iterations..

    Args:
        G (DirectedGraph): A directed graph.
        num_iter (int): Number of iterations for the PageRank computation. If 0, assigns equal weight to all nodes.
        eps (float): Ensures that nodes are reachable evenif no incoming edges exist. Important for nice mathematical properties.
    """

    n = G.number_of_nodes()
    scores = np.array([1/n for i in range(n)])

    if num_iter == 0:
        return scores

    # Compute out-degrees for each node
    out_degrees = np.array([sum(G.adj_matrix[node]) for node in range(n)])
    
    if G.adj_matrix.dtype != np.bool:
        sinks = []
        for i in range(n):
            if len(G.edges_from(i)) == 0:
                sinks += [i]

    # Iterative computation of PageRank scores using transposed graph
    for _ in range(num_iter):
        new_scores = np.zeros((n), dtype = np.float64)
        if G.adj_matrix.dtype != np.bool:
            sinks_score_sum = sum(scores[sinks])
        for v in range(n):
            incoming_score = sum((scores[v0] * G.get_edge(v0, v)) / out_degrees[v0] for v0 in np.where(G.adj_matrix.T[v] > 0)[0])
            new_scores[v] = eps/n + (1 - eps) * (incoming_score + (0 if G.adj_matrix.dtype == np.bool else sinks_score_sum * (1 / n)))

        scores = new_scores

    return scores

def visualize_page_rank(scores: np.ndarray, title: str):
    """
    Visualizes the PageRank scores as a bar chart.
    
    Args:
        scores (np.ndarray): A 1D array of PageRank scores, where each index represents a node.
        title (str): Title for the chart.
    """
    n = len(scores)
    nodes = np.arange(n)  # List of nodes from 0 to n-1

    # Create a bar plot of PageRank scores
    plt.figure(figsize=(10, 6))
    plt.bar(nodes, scores, color='skyblue')
    
    plt.xlabel("Node")
    plt.ylabel("PageRank Score")
    plt.title(title)
    plt.xticks(nodes)
    
    # Show the plot
    plt.savefig(title)



def create_combined_graph(body_path: str, title_path: str, indexing_path: str):
    body_df = pd.read_csv(body_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1], # Notice how I limited myself to certain columns, this make things much more efficient.
                          delimiter="\t")
    title_df = pd.read_csv(title_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1],
                          delimiter = "\t")
    combined_df = pd.concat([body_df, title_df], axis=0, ignore_index=True)
    title_df, body_df = None, None # I do this to free memory. 
    
    indexing_df = pd.read_csv(indexing_path,
                              usecols = [1])
                              
    indexing = {subreddit: idx for idx, subreddit in indexing_df.itertuples(name = None)}

    indexing_df = None
    
    combined_df['SOURCE_SUBREDDIT'] = combined_df['SOURCE_SUBREDDIT'].map(indexing)
    combined_df['TARGET_SUBREDDIT'] = combined_df['TARGET_SUBREDDIT'].map(indexing)

    weighted_graph = DirectedGraph(len(indexing), weighted  = True)
    unweighted_graph = DirectedGraph(len(indexing), weighted = False)
    printed = nx.DiGraph()

    for source_target, df in combined_df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']):
        weighted_graph.add_edge(source_target[SOURCE_SUBREDDIT], source_target[TARGET_SUBREDDIT], weight = df.shape[NUM_ROWS])
        printed.add_edge(source_target[SOURCE_SUBREDDIT], source_target[TARGET_SUBREDDIT], weight = df.shape[NUM_ROWS])
        unweighted_graph.add_edge(source_target[SOURCE_SUBREDDIT], source_target[TARGET_SUBREDDIT])

    pos = nx.spring_layout(printed)  # Layout for the nodes
    nx.draw(printed, pos, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
    nx.draw_networkx_edge_labels(printed, pos, edge_labels=weighted_graph)

    plt.title("Directed Graph Visualization")
    plt.savefig("Directed Graph Visualization")
    
    return weighted_graph, unweighted_graph

def main():
    weighted_graph, unweighted_graph = create_combined_graph(body_path = BODY_DS_PATH ,title_path = BODY_DS_PATH, indexing_path = SUBREDDIT_INDEXING_PATH)
    influence_weighted = page_rank_plus(weighted_graph, num_iter = 10)
    influence_unweighted = page_rank_plus(unweighted_graph, num_iter = 10)
    visualize_page_rank(influence_weighted, title="PageRank plus Scores")
    visualize_page_rank(influence_unweighted, title="PageRank Scores")
    

if __name__ == "__main__":
    main()

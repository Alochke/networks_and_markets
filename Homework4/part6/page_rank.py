import numpy as np
from typing import Union
import pandas as pd

TITLE_DS_PATH = "soc-redditHyperlinks-body.tsv"
BODY_DS_PATH = "soc-redditHyperlinks-title.tsv"
SUBREDDIT_INDEXING_PATH = "subbreddit_indexing.csv"

##########################################################################################
class DirectedGraph:
    def __init__(self, number_of_nodes, weighted : bool):
        self.adj_matrix: np.ndarray = np.zeros((number_of_nodes, number_of_nodes), dtype=np.int32 if weighted  else np.bool)
    
    def add_edge(self, origin_node, destination_node, weight = True):
        '''Adds an edge from origin_node to destination_node.'''
        self.adj_matrix[origin_node][destination_node] = weight
    
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
    out_degrees = np.array(sum(G.edges_from(node) for node in range(n)))
    
    if G.adj_matrix.dtype == np.int32:
        num_sinks = len(np.where(G.adj_matrix == 0)[0])

    # Iterative computation of PageRank scores using transposed graph
    for _ in range(num_iter):
        new_scores = np.zeros((n), dtype = np.float32)
        for v in range(n):
            incoming_score = sum((scores[v0] * G.get_edge(v0, v)) / out_degrees[v0] for v0 in transposed_G.edges_from(v))
            new_scores[v] = eps/n + (1 - eps) * (incoming_score + (0 if G.adj_matrix.dtype == np.bool else num_sinks * (1 / n)))

        scores = new_scores

    return scores

def create_combined_graph(body_path: str, title_path: str, indexing_path: str):
    body_df = pd.read_csv(body_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1], # Notice how I limited myself to certain columns, this make things much more efficient.
                          delimiter="\t")
    title_df = pd.read_csv(title_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1],
                          delimiter="\t")
    combined_df = pd.concat([body_df, title_df], axis=0, ignore_index=True)
    title_df, body_df = None, None # I do this to free memory. 
    
    indexing_df = pd.read_csv(indexing_path,
                              usecols = [1])
    indexing_df.
    indexing = {subreddit: idx for idx, subreddit in enumerate(indexing_df[0])}

    indexing_df = None
    combined_df['SOURCE_SUBREDDIT'] = combined_df['SOURCE_SUBREDDIT'].map(indexing)
    combined_df['TARGET_SUBREDDIT'] = combined_df['TARGET_SUBREDDIT'].map(indexing)

    weighted_graph = DirectedGraph(len(indexing), weighted  = True)
    unweighted_graph = DirectedGraph(len(indexing), weighted = False)

    for source_target, df in combined_df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']):
        weighted_graph.add_edge(source_target[0], source_target[1], weight = df.shape()[0])
        unweighted_graph.add_edge(source_target[0], source_target[1])
    
    return weighted_graph, unweighted_graph

def main():
    weighted_graph, unweighted_graph = create_combined_graph(body_path = BODY_DS_PATH ,title_path = BODY_DS_PATH, indexing_path = SUBREDDIT_INDEXING_PATH)
    """love you bro"""
    print(weighted_graph)
    print(unweighted_graph)
if __name__ == "__main__":
    main()

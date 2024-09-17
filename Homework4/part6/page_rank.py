import numpy as np
import os
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
##########################################################################################
class DirectedGraph:
    def __init__(self, number_of_nodes, wheighted: bool):
        self.adj_matrix: np.ndarray = np.zeros((number_of_nodes, number_of_nodes), dtype=np.int32 if wheighted else np.bool)
    
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


    #loading graphs in part 6
######################################################################################
def create_graph_from_dataframe(df, subreddit_to_number, weighted=False):
    '''
    Convert subreddit names to numbers and create a graph based on a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns 'SOURCE_SUBREDDIT' and 'TARGET_SUBREDDIT'.
        subreddit_to_number (dict): Mapping from subreddit names to numbers.
        weighted (bool): If True, the graph will consider weights; otherwise, it will use binary connections.

    Returns:
        DirectedGraph: The graph constructed from the DataFrame.
    '''
    # Convert subreddit names to numbers

    df['source_number'] = df['SOURCE_SUBREDDIT'].map(subreddit_to_number)
    df['target_number'] = df['TARGET_SUBREDDIT'].map(subreddit_to_number)
   
    # Initialize the graph
    graph = DirectedGraph(len(subreddit_to_number), weighted)

    # Add edges to the graph
    for _, row in df.iterrows():
        graph.add_edge(row['source_number'], row['target_number'])

    return graph

def main():
    """
    TEST TO CHECK HOW THE GRAPH TO CREATE THE GRAPH FROM THE TSV FILES
    
    ALON CHANGRE YOUR PATHSSSS HEREEEEE
    """
    subbredit_title_to_number_path = 'unique_subreddits_title.csv'
    title_subreddit_graph_path = 'soc-redditHyperlinks-title.tsv'

    subbredit_title_to_number_df = pd.read_csv(subbredit_title_to_number_path)

    title_subreddit_graph = pd.read_csv(title_subreddit_graph_path,sep='\t')
    subreddit_to_number = {subreddit: idx for idx, subreddit in enumerate(subbredit_title_to_number_df['Subreddit'])}

    graph = create_graph_from_dataframe(df=title_subreddit_graph,subreddit_to_number=subreddit_to_number,weighted=False)


if __name__ == "__main__":
    main()
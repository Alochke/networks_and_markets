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
def uniform_dataframe_vertically(df1,df2):
    union = pd.concat([df1, df2], axis=0, ignore_index=True)
    return union

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


   
    # Initialize the graph
    graph = DirectedGraph(len(subreddit_to_number), weighted)

    # Add edges to the graph
    if graph.adj_matrix.dtype == np.int32: #created weighted for how much time the edge (x,y) has apeeared in the file
        edge_counts = df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).size().reset_index(name='COUNT')
        
        edge_counts['source_number'] = df['SOURCE_SUBREDDIT'].map(subreddit_to_number)
        edge_counts['target_number'] = df['TARGET_SUBREDDIT'].map(subreddit_to_number)
        for _, row in edge_counts.iterrows():
            graph.add_edge(row['source_number'], row['target_number'],weight=row['COUNT'])

    else:
        df['source_number'] = df['SOURCE_SUBREDDIT'].map(subreddit_to_number)
        df['target_number'] = df['TARGET_SUBREDDIT'].map(subreddit_to_number)
        for _, row in df.iterrows():
            graph.add_edge(row['source_number'], row['target_number'])


    return graph


def combined_body_title_graph(body_df,title_df,subreddit_to_number,weighted):
    combined_df = uniform_dataframe_vertically(df1=body_df,df2=title_df)
    graph = create_graph_from_dataframe(df = combined_df , subreddit_to_number=subreddit_to_number,weighted=weighted )
    return graph



def create_combined_graph(body_path,title_path,combined_unique_path,weighted):
    body_df = pd.read_csv(body_path,sep = "\t")
    title_df = pd.read_csv(title_path,sep = "\t")
    comdined_subreddit_to_number_df = pd.read_csv(combined_unique_path)
    comdined_subreddit_to_number  =  {subreddit: idx for idx, subreddit in enumerate(comdined_subreddit_to_number_df['Subreddit'])}
    graph = combined_body_title_graph(body_df=body_df,title_df=title_df,subreddit_to_number=comdined_subreddit_to_number,weighted=weighted)
    return graph   

def main():
    title_df_path = "/Users/arielchiskis/Downloads/soc-redditHyperlinks-title.tsv"
    body_df_path = "/Users/arielchiskis/Downloads/soc-redditHyperlinks-body.tsv"
    combined_unique_subreddit_path = "networks_and_markets/Homework4/part6/unique_subreddits_combined_title_and_body.csv" #relative path king
    weighted = True
    graph = create_combined_graph(body_path=body_df_path,title_path=title_df_path,combined_unique_path=combined_unique_subreddit_path,weighted=weighted)
    """continue from here alon ya king"""
 
    print("finished")
if __name__ == "__main__":
    main()

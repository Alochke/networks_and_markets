import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Tuple
import os

TITLE_DS_PATH = "soc-redditHyperlinks-body.tsv"
BODY_DS_PATH = "soc-redditHyperlinks-title.tsv"
SUBREDDIT_INDEXING_PATH = "subbreddit_indexing.csv"
SOURCE_SUBREDDIT = 0
TARGET_SUBREDDIT = 1
NUM_ROWS = 0

GET_NAME = lambda indexing_df, i: indexing_df.iat[i, 0]

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
    
    def get_in_degree(self, node):
        '''Return the in-degree of node.'''
        return len(np.where(self.transpose()[node] > 0)[0])



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

def visualize_pagerank_results(pagerank_results, title):
    # Create the directory for storing results if it doesn't exist
    results_dir = "pagerank_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Prepare data for plotting
    nodes = [i for i in range(len(pagerank_results))]

    # Create a bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(nodes, pagerank_results, color='skyblue')
        
    plt.xlabel('Node')
    plt.ylabel('Score')
    plt.title(title)

    # Save the plot to the designated folder
    plt.savefig(f"{results_dir}/{title.replace(' ', '_')}.png")
    plt.close()  # Close the figure context to free up memory


def create_combined_graph(body_path: str, title_path: str, indexing_df: pd.DataFrame) -> Tuple[DirectedGraph, DirectedGraph]:
    body_df = pd.read_csv(body_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1],
                          delimiter="\t")
    title_df = pd.read_csv(title_path, dtype = {"SOURCE_SUBREDDIT": str, "TARGET_SUBREDDIT": str},
                          usecols = [0, 1],
                          delimiter = "\t")
    combined_df = pd.concat([body_df, title_df], axis=0, ignore_index=True)
    title_df, body_df = None, None # I do this to free memory. 
                              
    indexing = {subreddit: idx for idx, subreddit in indexing_df.itertuples(name = None)}

    indexing_df = None
    
    combined_df['SOURCE_SUBREDDIT'] = combined_df['SOURCE_SUBREDDIT'].map(indexing)
    combined_df['TARGET_SUBREDDIT'] = combined_df['TARGET_SUBREDDIT'].map(indexing)

    weighted_graph = DirectedGraph(len(indexing), weighted  = True)
    unweighted_graph = DirectedGraph(len(indexing), weighted = False)

    for source_target, df in combined_df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']):
        weighted_graph.add_edge(source_target[SOURCE_SUBREDDIT], source_target[TARGET_SUBREDDIT], weight = df.shape[NUM_ROWS])
        unweighted_graph.add_edge(source_target[SOURCE_SUBREDDIT], source_target[TARGET_SUBREDDIT])
    
    return weighted_graph, unweighted_graph

def run_alg_and_analyse(graph: DirectedGraph, analysed: str) -> np.ndarray:
    ANALYSIS_STRING = lambda x, t0, t1: f"Average infulence: {np.average(x)}\nStandard deviation: {np.std(x)}\nMinimal Influence: {min(x)}\nMaximal influence: {max(x)}\nSum of influences: {sum(x)}\nRuntime: {t1 - t0} seconds\n"
    
    t0 = time.monotonic()
    influence_weighted = page_rank_plus(graph, num_iter = 10)
    t1 = time.monotonic()
    
    print(f"Analysis of {analysed} algorithm\n")
    print(ANALYSIS_STRING(influence_weighted, t0, t1))
    
    return influence_weighted

def main():
    indexing_df = pd.read_csv(SUBREDDIT_INDEXING_PATH,
                              usecols = [1])
    weighted_graph, unweighted_graph = create_combined_graph(body_path = BODY_DS_PATH ,title_path = BODY_DS_PATH, indexing_df = indexing_df)
    
    print(f"Average in degree: {np.average(np.array([weighted_graph.get_in_degree(i) for i in range(weighted_graph.number_of_nodes())]))}")
    print(f"Number of sinks: {len([i for i in range(unweighted_graph.number_of_nodes()) if len(unweighted_graph.edges_from(i))])}\n")

    influence_unweighted = run_alg_and_analyse(unweighted_graph, "regular")
    influence_weighted = run_alg_and_analyse(weighted_graph, "improved")
    normalized_weighted = influence_weighted / max(influence_weighted)
    normalized_unweighted = influence_unweighted / max(influence_unweighted)
    visualize_pagerank_results(influence_unweighted, "Scaled PageRank Results")
    visualize_pagerank_results(influence_weighted, "Improved PageRank Results")

    abs_diff = np.abs(normalized_weighted - normalized_unweighted)
    biggest_diffrence = [pair[0] for pair in sorted([[i, abs_diff[i]] for i in range(len(abs_diff))], key = lambda x: x[1], reverse = True)][:10]

    print(f"10 nodes with biggest diffrence between their normalized pageRank score and normalized improved PageRank score: {[GET_NAME(indexing_df, i) for i in biggest_diffrence]}")
    print(f"Their in-degrees, respectively: {[weighted_graph.get_in_degree(i) for i in biggest_diffrence]}")
    print(f"Their normalized PageRank scores, respectively: {[normalized_unweighted[i] for i in biggest_diffrence]}")
    print(f"Their normalized Improved PageRank scores, respectively: {[normalized_weighted[i] for i in biggest_diffrence]}\n")

    highest_influence_unweighted = [pair[0] for pair in sorted([[i, normalized_unweighted[i]] for i in range(len(abs_diff))], key = lambda x: x[1], reverse = True)][:10]
    print(f"10 nodes with highest normal PageRank result: {[GET_NAME(indexing_df, i) for i in highest_influence_unweighted]}")
    print(f"Their normalized normal PageRank scores, respectively: {[normalized_unweighted[i] for i in highest_influence_unweighted]}\n")

    highest_influence_unweighted = [pair[0] for pair in sorted([[i, normalized_weighted[i]] for i in range(len(abs_diff))], key = lambda x: x[1], reverse = True)][:10]
    print(f"10 Nodes with highest improved PageRank result: {[GET_NAME(indexing_df, i) for i in highest_influence_unweighted]}")
    print(f"Their normalized improved PageRank scores, respectively: {[normalized_unweighted[i] for i in highest_influence_unweighted]}")

if __name__ == "__main__":
    main()

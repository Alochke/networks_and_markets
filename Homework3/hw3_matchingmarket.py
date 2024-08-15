# Skeleton file for HW3 questions 7 and 8
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before submission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

SEED = 42
RNG = np.random.default_rng(SEED)


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
        self.outcome = np.zeros(number_of_nodes, dtype=int)  # initialize all actions to Y
        self.final = False

    def add_edge(self, nodeA, nodeB):
        """
        Add an undirected edge to the graph between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        """
        self.adj_matrix[nodeA][nodeB] = 1
        self.adj_matrix[nodeB][nodeA] = 1
    
    def edges_from(self, nodeA: int):
        """
        Return a list of all nodes connected to nodeA by an edge.
        Args:
            nodeA (int): Index of the node to retrieve edges from.
        Returns:
            list[int]: List of nodes that have an edge with nodeA.
        """
        if not self.final:
            return list(np.where(self.adj_matrix[int(nodeA)] > 0)[0])
        else:
            return self.neighbors[int(nodeA)]
    
    def check_edge(self, nodeA, nodeB):
        """
        Check if there is an edge between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        Returns:
            bool: True if there is an edge, False otherwise.
        """
        return self.adj_matrix[nodeA][nodeB] > 0
    
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

    def finalize_neighbors(self):
        self.neighbors = [[] for _ in range(self.num_nodes)]
        for v in range(self.num_nodes):
            self.neighbors[v] = self.edges_from(v)
        self.final = True


class WeightedDirectedGraph(UndirectedGraph):
    def set_edge(self, origin_node, destination_node, weight=1):
        self.adj_matrix[origin_node][destination_node] = weight

    def get_edge(self, origin_node, destination_node):
        return self.adj_matrix[origin_node][destination_node]

    def neighbors(self, node):
        return [i for i in range(self.num_nodes) if self.adj_matrix[node][i] > 0]


def build_graph(n, m, V, P):
    """
    Build the graph for the market equilibrium problem, incorporating buyers' preferred choices
    based on their valuations minus current item prices.

    Parameters:
    n (int): Number of buyers.
    m (int): Number of items.
    V (list of lists): Valuations where V[i][j] is the valuation of buyer i for item j.
    P (list): Current prices of items.
    
    Returns:
    WeightedDirectedGraph: The constructed graph with buyers, items, source, and sink nodes.
    """
    num_nodes = n + m + 2  # +2 for source and sink
    graph = WeightedDirectedGraph(num_nodes)
    source = n + m  # Index of the source node
    sink = n + m + 1  # Index of the sink node

    # Connect source to each buyer
    for i in range(n):
        graph.set_edge(source, i, 1)  # Each buyer can "buy" one item

    # Connect each item to sink
    for j in range(m):
        graph.set_edge(n + j, sink, 1)  # Each item can be "sold" once

    # Connect each buyer to items based on maximum adjusted valuation
    for i in range(n):
        max_utility = max(V[i][j] - P[j] for j in range(m))  # Find the highest utility for buyer i
        for j in range(m):
            if V[i][j] - P[j] == max_utility:  # Connect only to items offering max utility
                if (V[i][j] - P[j] == max_utility >= 0):
                    graph.set_edge(i, n + j, 1)  # Set edge capacity to 1

    return graph


def bfs_capacity(graph, source, sink, parent):
    """ Perform BFS to find a path with available capacity from source to sink """
    visited = [False] * graph.num_nodes
    queue = deque([source])
    visited[source] = True
    
    while queue:
        current = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            if not visited[neighbor] and graph.get_edge(current, neighbor) > 0:  # Capacity must be greater than 0
                queue.append(neighbor)
                visited[neighbor] = True
                parent[neighbor] = current
                if neighbor == sink:
                    return True
    return False


def max_flow(graph, source, sink):

    parent = [-1] * graph.num_nodes  # Array to store the path
    max_flow = 0
    
    # Augment the flow while there is a path from source to sink
    while bfs_capacity(graph, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph.get_edge(parent[s], s))
            s = parent[s]

        # Update residual capacities of the edges and reverse edges along the path
        v = sink
        while v != source:
            u = parent[v]
            graph.adj_matrix[u][v] -= path_flow
            if graph.adj_matrix[v][u] > 0:
                graph.adj_matrix[v][u] += path_flow
            else:
                graph.adj_matrix[v][u] = path_flow
            v = parent[v]

        max_flow += path_flow

    return max_flow, graph


def find_reachable_nodes(graph, source):
    visited = set()
    queue = deque([source])
    visited.add(source)
    
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if graph.get_edge(node, neighbor) > 0 and neighbor not in visited:  # Check residual capacity
                visited.add(neighbor)
                queue.append(neighbor)
                
    return visited


def find_constricted_set(graph, source, rightSide):
    reachable = find_reachable_nodes(graph, source)
    # print(reachable)
    constircted_set = [node for node in range(rightSide) if node in reachable and node != source]
    return constircted_set


def adjust_prices(graph, P, constricted_set, n):
    """
    Adjust prices based on the neighbors of the constricted set in the graph.
    Only the prices of items directly linked to the constricted set are increased.

    Parameters:
    graph (WeightedDirectedGraph): The graph used for max flow calculations.
    P (list): Current prices of items.
    constricted_set (list): List of buyers that form the constricted set.
    n (int): Number of buyers, used to differentiate buyer and item indices.
    """
    # Find items that are neighbors of the constricted buyers
    item_neighbors = set()
    for buyer_index in constricted_set:
        if buyer_index < n:  # Ensure it's a buyer index
            for item_index in graph.neighbors(buyer_index):
                if n <= item_index < n + len(P):  # Ensure it's within the item index range
                    item_neighbors.add(item_index - n)  # Adjust to item index based on 'n'

    # Increase prices for these items
    for item_index in item_neighbors:
        P[item_index] += 1  # Increase price by 1

    # Normalize prices to make the minimum price 0
    min_price = min(P)
    if min_price != 0:
        P = [price - min_price for price in P]  # Adjust all prices down by the minimum price

    return P  # Return the updated price list for clarity


def max_matching(n, m, graph):
    """finds max matching from a bipartite Preferred Choice graph of n nodes (buyer) in right side
    m nodes (items) in left side"""
 
    source = n+m
    sink = n+m+1

    # Calculate maximum flow in the graph
    max_flow_value, flow_graph = max_flow(graph, source, sink)

    # print(f"max flow: {max_flow_value}")
    # print("graph is:")

    # graph.print_graph()

    # print("flow_graph is:")
    # flow_graph.print_graph()
 
    # Extract matching from the flow graph
    matching = [None] * n
    for i in range(n):
        for j in range(m):
            if flow_graph.get_edge(j +n ,i) > 0:  # Check if there's positive flow from buyer i to item j
            #if in  the residual graph, in the max flow there is an edge from  j+m to i , then there is a match between
            # (i, j+m ) in ther graph !
                matching[i] = j # you can do also j + m if you want so it will represent the graph more precisely
                break

    return matching , flow_graph




# === Problem 7(a) ===
def matching_or_cset(n, C):
    # first create graph
    num_nodes = n + n + 2  # +2 for source and sink
    graph = WeightedDirectedGraph(num_nodes)
    source = n + n  # Index of the source node
    sink = n + n + 1  # Index of the sink node

    # Connect source to each buyer
    for i in range(n):
        graph.set_edge(source, i, 1)  # Each buyer can "buy" one item

    # Connect each item to sink
    for j in range(n):
        graph.set_edge(n + j, sink, 1)  # Each item can be "sold" once

    for i in range(n):
        for j in range(n):
            if C[i][j] != 0:  # Assuming non-zero entries indicate an edge
                graph.set_edge(i, n+j, C[i][j])

    M , residual_graph = max_matching(n=n, m=n, graph=graph)

    if (None in M):
        constricted_set = find_constricted_set(graph=residual_graph, source=source, rightSide=n)
        return (False,constricted_set)
    
    else:
        return (True,M)


# === Problem 7(b) ===

def market_eq(n, m, V):
    """
    Finds market equilibrium for a matching market with n buyers and m items.
    
    Parameters:
    n (int): Number of buyers.
    m (int): Number of items.
    V (list of lists): Buyer valuations, where V[i][j] is buyer i's value for item j.
    
    tuple: A tuple (P, M) of prices P and a matching M where P[j] is the price of item j,
           and M[i] = j if buyer i is matched with item j, M[i] = None if no matching for buyer i.
    """


    # Initial setup

  
    P = [0] * m  # Initial prices are set to 0 for all items.

    if (n <=m):
        
        source = n+m
        sink = n+m+1
        right_side = n
        left_side = m
        graph = build_graph(n=right_side, m=left_side, V=V, P=P)  # Function to build the initial graph based on valuations and prices

    else:
        # n > m
        #update V in all
        for i in range(n-m):
            for buyer in V:
                buyer.append(0)
            
            P.append(0)
        source = n + n
        sink = n + n + 1
        #now n = m
        right_side = n
        left_side = n
        graph = build_graph(n=right_side, m=left_side , V=V, P=P)  # Function to build the initial graph based on valuations and prices


    #graph.print_graph()
    while True:
        max_flow_value, residual_graph = max_flow(graph, source, sink)
        #(max_flow_value)
        #residual_graph.print_graph()
        if max_flow_value == right_side:  # Assuming supply equals demand exactly
            break  # We found a perfect matching, hence market equilibrium
        constricted_set = find_constricted_set(graph=residual_graph, source=source, rightSide=n)

        adjust_prices(residual_graph,P,constricted_set,n)  # Function to adjust prices based on the constricted set
        graph = build_graph(n=right_side, m=left_side, V=V, P=P)  # Rebuild the graph with updated prices


    graph = build_graph(right_side, left_side, V, P)
    M, _ = max_matching(right_side,left_side,graph)

    if (n > m):
        #right_side = n
        #left
        for i in range(n):
            if M[i] >= m:
                M[i] = None  #delete the imagenary matches of the imaginary items


        P = P[:m] # delete the imagenary prices of the imaginary items

    return P, M


# === Problem 8(b) ===
def vcg(n, m, V):
    '''Given a matching market with n buyers, and m items, and
    valuations V as defined in market_eq, output a tuple (P,M)
    of prices P and a matching M as computed using the VCG mechanism
    with Clarke pivot rule.
    V,P,M are defined equivalently as for market_eq. Note that
    P[j] should be positive for every j in 0...m-1. Note that P is
    still indexed by item, not by player!!
    '''
    # P = [0]*m
    # M = [0]*n

    _, M = market_eq(n, m, V)
    P = [0] * m
    SV_M = sum(V[j][M[j]] for j in range(n) if M[j] is not None)
    for i in range(n):
        # get best matching without 𝑖
        V_no_i = V[:i] + V[i+1:]
        _, M_no_i = market_eq(n-1, m, V_no_i)
        # update the price of item M[i] as the price buyer i should pay for the item it's matched to
        if M[i] is not None:
            P[M[i]] = sum(V_no_i[j][M_no_i[j]] for j in range(n-1) if M_no_i[j] is not None) - (SV_M - V[i][M[i]])
    return (P,M)


# === Bonus Question 2(a) (Optional) ===
def random_bundles_valuations(n, m):
    '''Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player'''
    V = [[0] * m for _ in range(n)]

    # Generate random valuations
    for i in range(n):
        # Sample a random value for the good for buyer i
        value_per_good = np.random.randint(1, 51)
        for j in range(m):
            # The valuation for bundle j is value_per_good multiplied by j
            V[i][j] = value_per_good * j

    return (n,m,V)


def run_vcg_on_random_bundles_valuations():
    n, m, V = random_bundles_valuations(20, 20)
    P, M = vcg(n, m, V)
    print("\nBonus 2(a), random bundles valuations:")
    print("Buyers value per good: ", prices_to_string([V[i][1] for i in range(n)]))
    print("Matching: ", matching_to_string(M))
    print("Item prices: ", prices_to_string(P))
    print("_____________________________________________________________________")


def random_bundles_valuations_sorted(n, m):
    '''Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player.
    The buyers are sorted in ascending order by their value per good
    s.t buyer n has the highest value per good.'''

    # Generate random values per good for each buyer
    values_per_good = np.random.randint(1, 51, size=n)
    sorted_values_per_good = sorted(values_per_good)

    # Initialize and assign valuations matrix V based on sorted values per good
    V = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            # The valuation for bundle j is value_per_good multiplied by j
            V[i][j] = sorted_values_per_good[i] * j

    return (n,m,V)


def run_vcg_on_random_bundles_valuations_sorted():
    n, m, V = random_bundles_valuations_sorted(20, 20)
    P, M = vcg(n, m, V)
    print("\nBonus 2(a), random bundles valuations (sorted):")
    print("Buyers are sorted in the ascending order of their value per good.")
    print("Buyers value per good: ", prices_to_string([V[i][1] for i in range(n)]))
    print("Matching: ", matching_to_string(M))
    print("Item prices: ", prices_to_string(P))
    externality_p = [0] * m
    for i in range(n):
        # if i doesn't play, each player j s.t j<i (ascending order) will be assigned a bundle that is bigger by 1 item
        # V[j][1] is the valuation of buyer j for 1 good
        externality_p[i] = sum(V[j][1] for j in range(i))
    print("Externality prices: ", prices_to_string(externality_p))
    print("Price of bundle divided by the number of items in the bundle:")
    print([f"{P[j]/j:.2f}" for j in range(1, m)])       # p(y_j)/c_j)
    print("_____________________________________________________________________")


# === Bonus Question 2(b) (optional) ===
def gsp(n, m, V):
    '''Given a matching market for bundles with n buyers, and m bundles, and
    valuations V (for bundles), output a tuple (P, M) of prices P and a 
    matching M as computed using GSP.'''
    P = [0]*m
    _, M = market_eq(n, m, V)

    # Extract the valuations for bundle 1 (value per 1 good)
    valuations = [V[i][1] for i in range(n)]

    # Get the indices that would sort the valuations in ascending order
    sorted_indices = np.argsort(valuations)

    # Reorder the player indices based on the sorted order
    sorted_V = [V[i] for i in sorted_indices]
    P[0] = 0

    for i in range(1, n):
        P[i] = i * sorted_V[i-1][1]  # Use sorted valuations to set prices
        # "next lower bid" = "second price" (bids sorted in ascending order)

    # Translate to original buyer indices
    # Map the prices back to original indices
    original_indices = np.argsort(sorted_indices)  # Reverse mapping to original indices

    P_original = [0] * n
    for idx, orig_idx in enumerate(original_indices):
        P_original[orig_idx] = P[idx]

    return (P_original,M)


def compare_vcg_gsp(n, m, V):
    P_vcg, M_vcg = vcg(n, m, V)
    P_gsp, M_gsp = gsp(n, m, V)
    print("Buyers value per good: ", [int(V[i][1]) for i in range(n)])
    print("VCG:")
    print("Matching: ", M_vcg)
    print("Item prices: ")
    print_floats_list(P_vcg)
    print("GSP:")
    print("Matching: ", M_gsp)
    print("Item prices: ")
    print_floats_list(P_gsp)
    print("Item prices GSP - VCG: ")
    print_floats_list([P_gsp[i] - P_vcg[i] for i in range(m)])
    print("_____________________________________________________________________")


def compare_vcg_gsp_on_random_bundles():
    n, m, V = random_bundles_valuations_sorted(20, 20)
    print("\nBonus 2(b), VCG and GSP comparison for random bundles valuations (sorted):")
    print("Buyers are sorted in the ascending order of their value per good.")
    compare_vcg_gsp(n, m, V)


def compare_vcg_gsp_on_constant_bundles():
    n = 20
    m = 20
    c = 2   # constant price of c per good for all players
    V = [[c*i for i in range(m)] for _ in range(n)]
    print("\nBonus 2(b), VCG and GSP comparison for bundles with constant valuations:")

    compare_vcg_gsp(n, m, V)


#Test for 7(a)
def positive_test():
    # Example connectivity matrix, change as needed for different scenarios
    C = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Number of vertices in each part of the bipartite graph
    n = 4

    # Assuming matching_or_cset and other required functions are defined and ready to use
    result = matching_or_cset(n, C)

    # Output the results
    print("Is there a perfect matching? ", result[0])
    print("Matching or Constricted Set: ", result[1])

def negative_test():
    # Example connectivity matrix where a perfect matching is not possible
    C = [
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ]

    # Number of vertices in each part of the bipartite graph
    n = 4

    # Assuming matching_or_cset and other required functions are defined and ready to use
    result = matching_or_cset(n, C)

    # Output the results
    print("Is there a perfect matching? ", result[0])
    print("Matching or Constricted Set: ", result[1])

def question2_test():

    # Define the number of buyers and items
    n = 3  # number of buyers
    m = 3  # number of items

    # Define the valuations matrix as given in the table
    V = [
        [5, 2, 5],  # Valuations by buyer 'a'
        [7, 3, 4],  # Valuations by buyer 'b'
        [1, 1, 4]   # Valuations by buyer 'c'

    ]

    V2 = [
        [5, 7, 1],
        [2, 3, 1],
        [5, 4, 4]

    ]

    print("\nTesting on q.2:")
    print("\nResults for the procedure constructed in Theorem 8.2:")
    # Run the market equilibrium algorithm
    P, M = market_eq(n, m, V2)

    print("Prices for items x, y, z:", P)
    print("Matching result (buyer to item):", M)

    print("\nResults for the VCG algorithm:")
    # run the VCG algorithm
    P, M = vcg(n, m, V2)
    print("Prices for items x, y, z:", P)
    print("Matching result (buyer to item):", M)
    print("_____________________________________________________________________")


def matching_to_string(M):
    return ", ".join([f"M({i})={M[i]}" for i in range(len(M))])


def prices_to_string(p):
    return ", ".join([f"p({i})={p[i]}" for i in range(len(p))])

def print_floats_list(float_list, decimals=2):
    formatted_floats = [f"{num:.{decimals}f}" for num in float_list]
    print("[" + ", ".join(formatted_floats) + "]")

def are_identical_lists(list1, list2):
    return len(list1) == len(list2) and all(x == y for x, y in zip(list1, list2))


def run_test(name, n, m, V):
    print("\nTesting on " + name + ":")
    print("\nResults for the procedure constructed in Theorem 8.2:")
    P_8_2, M_8_2 = market_eq(n, m, V)
    print("Matching: ", matching_to_string(M_8_2))
    print("Item prices: ", prices_to_string(P_8_2))
    print("\nResults for the VCG algorithm:")
    P_vcg, M_vcg = vcg(n, m, V)
    print("Matching: ", matching_to_string(M_vcg))
    print("Item prices: ", prices_to_string(P_vcg))
    print("_____________________________________________________________________")
    if are_identical_lists(M_8_2, M_vcg) and are_identical_lists(P_8_2, P_vcg):
        return True
    print("******")
    return False


def lec5_p7_test():
    n = 3
    m = 3

    V = [
        [4, 12, 5],
        [7, 10, 9],
        [7, 7, 10]
    ]

    run_test("the matching market frame with values as described in Lecture 5 Page 7", n, m, V)


def question1_test():
    n = 2
    m = 2

    V = [
        [2, 4],
        [3, 6],
    ]

    run_test("q.1", n, m, V)


def random_test():
    no_runs = 100
    n = 20
    m = 20

    identical_results_counter = 0

    for _ in range(no_runs):
        V = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                V[i][j] = np.random.randint(1, 51)

        iden = run_test("random", n, m, V)
        identical_results_counter += iden
    print(f"Identical results in {no_runs} runs: {identical_results_counter}")



def n_gt_m_test():
    n = 3
    m = 2

    V = [
        [4, 12],
        [7, 10],
        [7, 7]
    ]

    run_test("a matching market with n>m", n, m, V)

def random_test_a():
    n = RNG.choice(40)
    m = RNG.choice(40)

    V = [[RNG.choice(40) for i in range(m)] for j in range(n)]

    run_test("a matching market with random parameters", n, m, V)


def main():
    question2_test()
    lec5_p7_test()
    question1_test()
    n_gt_m_test()
    random_test()
    random_test_a()
    # bonus questions
    run_vcg_on_random_bundles_valuations()
    run_vcg_on_random_bundles_valuations_sorted()
    compare_vcg_gsp_on_random_bundles()
    compare_vcg_gsp_on_constant_bundles()


if __name__ == "__main__":
    main()





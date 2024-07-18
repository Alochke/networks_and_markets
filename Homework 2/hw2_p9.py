# Skeleton file for HW2 question 9
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
import seaborn as sns

FB_GRAPH_SIZE = 4039
FB_GRAPH_NUM_PAIRS = (FB_GRAPH_SIZE * (FB_GRAPH_SIZE - 1)) / 2
SEED = 42
DEBUG = False

# player actions
X = 1
Y = 0

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
        self.outcome = np.zeros(number_of_nodes, dtype=int)  # initialize all actions to Y
        self.final = False
        self.neighbors = [[] for _ in range(number_of_nodes)]

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
        for v in range(self.num_nodes):
            self.neighbors[v] = self.edges_from(v)
        self.final = True

def create_fb_graph(filename = "facebook_combined.txt"):
    ''' This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes.'''    
    res = UndirectedGraph(FB_GRAPH_SIZE)

    try:
        with open(filename) as f:
            for line in f:
                res.add_edge(*(int(node) for node in line.split()))
    except Exception as e:
        print(f"File related error: {e}")
        exit()

    res.finalize_neighbors()   # avoid recalculating neighbors
    return res


# === Problem 9(a) ===

def contagion_brd(G, S, t):
    '''Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
       and a float threshold t, perform BRD as follows:
       - Permanently infect the nodes in S with X
       - Infect the rest of the nodes with Y
       - Run BRD on the set of nodes not in S
       Return a list of all nodes infected with X after BRD converges.'''
    def should_defect(action, p):
        return (action == Y and p >= t) or (action == X and p < t)

    def get_defector():
        for v in range(G.num_nodes):
            if v not in S:
                neighbors = G.edges_from(v)
                if not neighbors:
                    continue
                neighbors_X = np.sum(G.outcome[neighbors])  # X = 1, Y = 0
                if should_defect(G.outcome[v], neighbors_X / len(neighbors)):
                    return v
        return None     # no defectors found

    # permanently infect adopters in S with X
    G.outcome[S] = X

    # infect the rest of the nodes with Y
    G.outcome[np.setdiff1d(np.arange(G.num_nodes), S)] = Y

    # run BRD on the set of nodes not in S
    while defector := get_defector():
        G.outcome[defector] = 1 - G.outcome[defector]

    # return a list of all nodes infected with X after BRD converges.
    # convert np.int64 to standard python int for submission
    return [int(v) for v in np.where(G.outcome == X)[0]] if not DEBUG else list(np.where(G.outcome == X)[0])


def test_contagion_brd():
    #   0 - 1 - 2 - 3
    graph_fig4_1_left = UndirectedGraph(4)
    graph_fig4_1_left.add_edge(0,1)
    graph_fig4_1_left.add_edge(1, 2)
    graph_fig4_1_left.add_edge(2, 3)
    print("\n === fig4_1_left === ")
    for t in np.arange(0, 0.75, 0.05):
        print(f"t={t:.2f}: ", contagion_brd(graph_fig4_1_left, [0, 1], t))

    #       2   4   6
    #       |   |   |
    #   0 - 1 - 3 - 5
    graph_fig4_1_right = UndirectedGraph(7)
    graph_fig4_1_right.add_edge(0,1)
    graph_fig4_1_right.add_edge(1, 2)
    graph_fig4_1_right.add_edge(1, 3)
    graph_fig4_1_right.add_edge(3, 4)
    graph_fig4_1_right.add_edge(3, 5)
    graph_fig4_1_right.add_edge(5, 6)
    print("\n === fig4_1_right === ")
    for t in np.arange(0, 0.75, 0.05):
        print(f"t={t:.2f}: ", contagion_brd(graph_fig4_1_right, [0, 1, 2], t))

def q_completecascade_graph_fig4_1_left():
    '''Return a float t s.t. the left graph in Figure 4.1 cascades completely.'''
    return 0.25      # threshold = 1/2

def q_incompletecascade_graph_fig4_1_left():
    '''Return a float t s.t. the left graph in Figure 4.1 does not cascade completely.'''
    0.6

def q_completecascade_graph_fig4_1_right():
    '''Return a float t s.t. the right graph in Figure 4.1 cascades completely.'''
    return 0.3      # threshold = 1/3

def q_incompletecascade_graph_fig4_1_right():
    '''Return a float t s.t. the right graph in Figure 4.1 does not cascade completely.'''
    0.4

def print_debug(str):
    if DEBUG:
        print(str)

def run_contagion_brd(G, k, t, n_iterations):
    '''k is the number of early adopters
       n_iterations is the number of runs'''
    infected = []
    rng = np.random.default_rng(SEED)
    for i in range(n_iterations):
        early_adopters = rng.choice(np.arange(FB_GRAPH_SIZE), size=k, replace=False)
        # print_debug(f"t={t}, k={k}, iteration {i}: Early adopters: {early_adopters}")
        cur_infected = contagion_brd(G, early_adopters, t)
        # print_debug(f"t={t}, k={k}, iteration {i}: Infected nodes: {cur_infected}")
        infected.append(len(cur_infected))
    return infected

# plots for 9c
def plot_surface(infection_rates, z_label, title):
    '''infection_rates is a list of (t, k, avg_infection) triplets'''
    t_values = sorted(set(t for t, k, _ in infection_rates))
    k_values = sorted(set(k for t, k, _ in infection_rates))

    T, K = np.meshgrid(t_values, k_values)
    Z = np.zeros(T.shape)

    for t, k, avg_infected in infection_rates:
        t_idx = t_values.index(t)
        k_idx = k_values.index(k)
        Z[k_idx, t_idx] = avg_infected

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, K, Z, cmap='viridis')

    ax.set_xlabel('Threshold t')
    ax.set_ylabel('Number of Early Adopters k')
    ax.set_zlabel(z_label)

    fig.colorbar(surf)
    plt.title(title)
    plt.show()

def plot_heatmap(infection_rates, title):
    t_values = sorted(set(t for t, k, _ in infection_rates))
    k_values = sorted(set(k for t, k, _ in infection_rates))

    Z = np.zeros((len(k_values), len(t_values)))

    for t, k, avg_infected in infection_rates:
        t_idx = t_values.index(t)
        k_idx = k_values.index(k)
        Z[k_idx, t_idx] = avg_infected

    plt.figure(figsize=(12, 8))
    sns.heatmap(Z, xticklabels=t_values, yticklabels=k_values, cmap='viridis', annot=True, fmt=".1f")
    plt.xlabel('Threshold t')
    plt.ylabel('Number of Early Adopters k')
    plt.title(title)
    plt.show()


def main():
    # === Problem 9(a) === #
    print("\n === Problem 9(a) === ")
    test_contagion_brd()
    # === Problem 9(b) === #
    print("\n === Problem 9(b) === ")
    fb_graph = create_fb_graph()
    if not DEBUG:
        n_iterations = 100
    else:
        n_iterations = 1
    infected = run_contagion_brd(fb_graph, 10, 0.1, n_iterations)
    print("nodes infected on average: ", np.mean(infected))
    print("variance of infected nodes: ", np.var(infected))
    print("number of cascades:", np.sum(np.array(infected) == fb_graph.number_of_nodes()), "in", n_iterations, "iterations")

    # === Problem 9(c) === #
    print("\n === Problem 9(c) === ")
    infection_rates = []
    cascades = []
    if not DEBUG:
        t_values = np.arange(0, 0.55, 0.05)
        k_values = np.arange(0, 251, 10)
        n_iterations = 10
    else:
        t_values = np.arange(0, 0.55, 0.05)
        k_values = np.arange(0, 41, 10)
        n_iterations = 1

    for t in t_values:
        for k in k_values:
            infected = run_contagion_brd(fb_graph, k, t, n_iterations)
            avg_infected = np.mean(infected)
            infection_rates.append((float(t), int(k), float(avg_infected)))
            cascades.append((float(t), int(k), float(np.sum(np.array(infected) == fb_graph.number_of_nodes()))))
            print_debug(f"{infection_rates[-1]}")

    # plot infections
    plot_surface(infection_rates, 'Average Infected', 'Average Infected Nodes')
    plot_heatmap(infection_rates, 'Average Infected Heatmap')

    # plot cascades
    plot_surface(cascades, 'Number of Cascades', 'Number of Cascades')
    plot_heatmap(cascades, 'Number of Cascades Heatmap')

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

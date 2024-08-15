# Skeleton file for HW3 questions 9 and 10
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
from hw3_matchingmarket import market_eq

SEED = 42
NUMBER_OF_TESTS = 100
RNG = np.random.default_rng(SEED)

# === Problem 9(a) ===
def exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs):
    '''Given a market scenario for ridesharing, with n riders, and
    m drivers, output an exchange network representing the problem.
    -   The grid is size l x l. Points on the grid are tuples (x,y) where
        both x and y are in (0...l).
    -   rider_vals is a list of numbers, where rider_vals[i] is the value
        of rider i's trip
    -   rider_locs is a list of points, where rider_locs[i] is the current
        location of rider i (in 0...n-1)
    -   rider_dests is a list of points, where rider_dests[i] is the desired
        destination of rider i (in 0...n-1)
    -   driver_locs is a list of points, where driver_locs[j] is the current
        location of driver j (in 0...m-1)
    - Output a tuple (n, m, V) representing a bipartite exchange network, where:
    V is an n x m list, with V[i][j] is the value of the edge between
    rider i (in 0...n-1) and driver j (in 0...m-1)'''
    L1_norm = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
    non_neg = lambda x: x if x >= 0 else 0
    V = [[non_neg(rider_vals[i] - (L1_norm(rider_locs[i], driver_locs[j]) + L1_norm(rider_locs[i], rider_dests[i]))) for j in range(m)] for i in range(n)]
    return (n, m, V)


# === Problem 10 ===
def stable_outcome(n, m, V):
    '''Given a bipartite exchange network, with n riders, m drivers, and
    edge values V, output a stable outcome (M, A_riders, A_drivers).
    -   V is defined as in exchange_network_from_uber.
    -   M is an n-element list, where M[i] = j if rider i is 
        matched with driver j, and M[i] = None if there is no matching.
    -   A_riders is an n-element list, where A_riders[i] is the value
        allocated to rider i.
    -   A_drivers is an m-element list, where A_drivers[j] is the value
        allocated to driver j.'''
    A_drivers, M = market_eq(n, m, V) 
    return (M, [V[i][M[i]] - A_drivers[M[i]] if (M[i] != None) else 0 for i in range(n)], A_drivers)

# === Problem 10(a) ===
def rider_driver_example_1():
    return (5, 7, 200, 
            [118, 666, 410, 122, 104], 
            [(187, 150), (182, 165), (27, 186), (168,  28), (195, 149)], 
            [(62, 27), (77, 181), (157,  45), (98, 170), (128,  61)],
            [(22, 193), (46, 103), (138,  64), (103,  56), (167, 121), (35, 66), (137, 135)])

def rider_driver_example_2():
    return (7, 5, 50, 
            [86, 34, 60, 48, 26, 52, 14], 
            [(24,  6), (12, 12), (1, 19), (36, 32), (38, 41), (3, 38), ( 5, 16)],
            [(4, 7), (24, 22), (17, 21), (21, 28), (31, 18), (37, 31), (24,  5)], 
            [(41, 11), (16, 38), (42, 49), (41, 40), (36, 42)])

# === Problem 10(b) ===
def random_riders_drivers_stable_outcomes(n, m):
    '''Generates n riders, m drivers, each located randomly on the grid,
    with random destinations, each rider with a ride value of 100, 
    and returns the stable outcome.'''
    GRID_SIZE = 100
    VALUE = 100
    ex_net = exchange_network_from_uber(n, m, GRID_SIZE, [VALUE] * n, RNG.choice(GRID_SIZE, size = (n, 2)), RNG.choice(GRID_SIZE, size = (n, 2)), RNG.choice(GRID_SIZE, size = (m, 2)))[2]
    return stable_outcome(n, m, ex_net)

# === Bonus 3(a) (Optional) ===
def public_transport_stable_outcome(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b):
    '''Given an l x l grid, n riders, m drivers, and public transportation
    parameters (a,b), output a stable outcome (M, A_riders, A_drivers), where:
    -   rider_vals, rider_locs, rider_dests, driver_locs are defined the same
        way as in exchange_network_from_uber
    -   the cost of public transport is a + b * dist(start, end) where dist is
        manhattan distance
    -   M is an n-element list, where M[i] = j if rider i is 
        matched with driver j, and M[i] = -1 if rider i takes public transportation, and M[i] = None if there is no match for rider i.
    -   A_riders, A_drivers are defined as before.
    -   If there is no stable outcome, return None.
    '''
    pass

def test_n_m(n: int, m: int):
    """
    Simulates multiple tests of rider and driver interactions to analyze their profit and price distributions.

    Parameters:
    n (int): The number of riders.
    m (int): The number of drivers.

    Returns:
    tuple: A tuple containing the following metrics:
        - Average rider social value (float)
        - Average rider profit (float)
        - Standard deviation of rider profits (float)
        - Minimum rider profit in all tests (int)
        - Maximum rider profit in all tests (int)
        - Average SV of the drivers (float)
        - Average price (float)
        - Standard deviation of prices (float)
        - Minimum price in all tests (int)
        - Maximum price in all tests (int)
    """
    rider_profit = []
    driver_value = []
    rider_value = []
    prices = []
    for i in range(NUMBER_OF_TESTS):
        M, A_riders, A_drivers = random_riders_drivers_stable_outcomes(n, m)
        rider_profit += A_riders
        driver_value.append(sum(A_drivers))
        rider_value.append(sum(A_riders))
        for i in range(len(A_drivers)):
            if any([j == i for j in M]):
                prices.append(A_drivers[i])
    return np.average(rider_value), np.average(rider_profit), np.std(rider_profit), min(rider_profit), max(rider_profit), np.average(driver_value), np.average(prices), np.std(prices), min(prices), max(prices)

def main():
    # === Problem 10(a) === #
    print("=== Problem 10(a) ===\n")

    print(f"example1:\n{rider_driver_example_1()}")
    print(f"example1 stable outcome:\n{stable_outcome(*exchange_network_from_uber(*rider_driver_example_1()))}\n")

    print(f"example2:\n{rider_driver_example_2()}")
    print(f"example2 stable outcome:\n{stable_outcome(*exchange_network_from_uber(*rider_driver_example_2()))}")
    

    # === Problem 10(b) === #
    N_INDX = 0
    M_INDX = 1
    NUM_STATS = 10

    print("\n=== Problem 10(b) ===")
    client_nums = [(10, 10), (20, 5), (5, 20)]
    # Stats is a lists such that in its cells are the following lists, respectively:
    # - A list where in cell i there is an approximation of the average rider sovial value for n riders and m drivers.
    # - A list where in cell i there is an approximation of the average rider profit for n riders and m drivers.
    # - A list where in cell i there is an approximation of the standard deviation of rider's profit for n riders and m drivers.
    # - A list where in cell i there is an approximation of the minimal rider profit for n riders and m drivers.
    # - A list where in cell i there is an approximation of the maximal rider profit for n riders and m drivers.
    # - A list where in cell i there is an approximation of the average driver social value for n riders and m drivers.
    # - A list where in cell i there is an approximation of the average price for n riders and m drivers.
    # - A list where in cell i there is an approximation of the standard deviation of a price for n riders and m drivers.
    # - A list where in cell i there is an approximation of the minimum price for n riders and m drivers.
    # - A list where in cell i there is an approximation of the maximum price for n riders and m drivers.
    # n and m are detrmined by client_nums, where in cell i, (n, m) resides.
    # I do it wierdly like this to eliminate code duplication.
    stats = [[] for i in range(NUM_STATS)]
    
    for n, m in client_nums:
            test_result = test_n_m(n, m)
            for i in range(len(stats)):
                stats[i].append(test_result[i])
    stats_names = ["Average rider social value", 
              "Average rider profits",
              "Standard deviation of rider profits",
              "Minimal rider profits",
              "Maximal rider profits",
              "Average driver social value", 
              "Average price",
              "Standard deviation of prices",
              "Minimal price",
              "Maximal price"]
    
    for i in range(len(client_nums)):
        print(f"\nPrinting calculations for number of riders = {client_nums[i][N_INDX]}, number of drivers = {client_nums[i][M_INDX]}\n")
        
        for j in range(NUM_STATS):
            print(f"{stats_names[j]}: {stats[j][i]}")

if __name__ == "__main__":
    main()

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
    Output a tuple (n, m, V) representing a bipartite exchange network, where:
    -   V is an n x m list, with V[i][j] is the value of the edge between
        rider i (in 0...n-1) and driver j (in 0...m-1)'''
    V = [[0] * m for _ in range(n)]
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
    M = [0] * n
    A_riders = [0] * n
    A_drivers = [0] * m
    return (M, A_riders, A_drivers)

# === Problem 10(a) ===
def rider_driver_example_1():
    # TODO fill in your own example
    return (n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)

def rider_driver_example_2():
    # TODO fill in your own example
    return (n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)

# === Problem 10(b) ===
def random_riders_drivers_stable_outcomes(n, m):
    '''Generates n riders, m drivers, each located randomly on the grid,
    with random destinations, each rider with a ride value of 100, 
    and returns the stable outcome.'''
    value = 100
    M = [0] * n
    A_riders = [0] * n
    A_drivers = [0] * m
    return (M, A_riders, A_drivers)

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


def main():
    # TODO: Put your analysis and plotting code here, if any
    print("hello world")

if __name__ == "__main__":
    main()

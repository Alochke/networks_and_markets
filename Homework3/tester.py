
from hw3_matchingmarket import matching_or_cset, market_eq, vcg
from hw3_uber import exchange_network_from_uber, stable_outcome, random_riders_drivers_stable_outcomes

# This file is provided for your convenience.
# These tests are by no means comprehensive, and are just a sanity check.
# Please write your own tests.


# === Problem 7(a) ===
# Graph from lecture 5 page 2
C = [[0] * 5 for _ in range(5)]
C[0][1] = 1
C[1][0] = 1
C[1][1] = 1
C[2][0] = 1
C[3][2] = 1
C[3][4] = 1
C[4][2] = 1
C[4][3] = 1
b, Cset = matching_or_cset(5, C)
assert b == False
assert set(Cset) == set([1, 0, 2]) # This should be the only constricted set in this example

# Graph from lecture 5 page 3
C = [[0] * 3 for _ in range(3)]
C[0][1] = 1
C[1][0] = 1
C[1][1] = 1
C[1][2] = 1
C[2][0] = 1
C[2][2] = 1
b, M = matching_or_cset(3, C)
assert b == True
assert (M == [1, 2, 0] or M == [1, 0, 2])

# === Problem 7(b) ===
# Context from lecture 5 page 7
V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
P, M = market_eq(3, 3, V)
assert P == [0,3,2]
assert M == [1,0,2]

# === Problem 8(a) ===
# Context from lecture 5 page 7
V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
P, M = vcg(3, 3, V)
assert P == [0,3,2]
assert M == [1,0,2]

# === Problem 9(a) ===
# Feed in some arbitrary numbers and check the values are positive and the same
n = 5
m = 5
l = 10
rider_vals = [100,100,100,100,100]
rider_locs = [(1,1),(1,1),(1,1),(1,1),(1,1)]
rider_dests = [(10,10),(10,10),(10,10),(10,10),(10,10)]
driver_locs = [(0,0),(0,0),(0,0),(0,0),(0,0)]
n, m, V = exchange_network_from_uber(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)
assert n == 5
assert m == 5
assert V[0][0] == V[4][4]
assert V[1][1] > 0

# === Problem 10 ===
# Context from lecture 5 page 7
V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
M, A_buyers, A_sellers = stable_outcome(3, 3, V)
assert M == [1,0,2]
assert A_sellers == [0,3,2]
assert A_buyers == [9,7,8]

print("Sanity Checks Passed.")

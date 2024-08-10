# Skeleton file for HW3 questions 7 and 8
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


# === Problem 7(a) ===
def matching_or_cset(n, C):
    '''Given a bipartite graph, with 2n vertices, with
    n vertices in the left part, and n vertices in the right part,
    and edge constraints C, output a tuple (b, M) where b is True iff 
    there is a matching, and M is either a matching or a constricted set.
    -   Specifically, C is a n x n array, where
        C[i][j] = 1 if left vertex i (in 0...n-1) and right vertex j (in 0...n-1) 
        are connected by an edge. If there is no edge between vertices
        (i,j), then C[i][j] = 0.
    -   If there is a perfect matching, return an n-element list M where M[i] = j 
        if driver i is matched with rider j.
    -   If there is no perfect matching, return a list M of left vertices
        that comprise a constricted set.
    '''
    isPerfect = False
    M = [0]*n
    return (isPerfect, M)

# === Problem 7(b) ===
def market_eq(n, m, V):
    '''Given a matching market with n buyers and m items, and 
    valuations V, output a market equilibrium tuple (P, M)
    of prices P and a matching M.
    -   Specifically, V is an n x m list, where
        V[i][j] is a number representing buyer i's value for item j.
    -   Return a tuple (P, M), where P is an m-element list, where 
        P[j] equals the price of item j.
        M is an n-element list, where M[i] = j if buyer i is 
        matched with item j, and M[i] = None if there is no matching.
    In sum, buyer i receives item M[i] and pays P[M[i]].'''
    P = [0]*m
    M = [0]*n
    return (P,M)

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
    P = [0]*m
    M = [0]*n
    return (P,M)


# === Bonus Question 2(a) (Optional) ===
def random_bundles_valuations(n, m):
    '''Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player'''
    V = [[0] * m for _ in range(n)]
    return (n,m,V)

# === Bonus Question 2(b) (optional) ===
def gsp(n, m, V):
    '''Given a matching market for bundles with n buyers, and m bundles, and
    valuations V (for bundles), output a tuple (P, M) of prices P and a 
    matching M as computed using GSP.'''
    P = [0]*m
    M = [0]*n
    return (P,M)


def main():
    # TODO: Put your analysis and plotting code here, if any
    print("hello world")

if __name__ == "__main__":
    main()


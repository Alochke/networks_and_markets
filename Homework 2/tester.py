import hw2_p9
import hw2_p10

# DO NOT SUBMIT THIS FILE
# It is just an example of a few tests that we will run on your code that you can use as a starting point
# to make sure the code is correct.
# You should put tester.py in the same folder as hw2_p9.py and hw2_p10.py, 
# as well as "facebook_combined.txt", and run this file.
# This is a sanity check and not representative of the final tests we will run on your submission.

# IMPORTANT
# We recommend that you write your own tests, in order to verify the correctness of your own code.
# This file is just a sanity check, and will not catch most bugs.


# Problem 9 sanity check
testGraph = hw2_p9.UndirectedGraph(3)
testGraph.add_edge(0, 1)
testGraph.add_edge(1, 2)
infected = hw2_p9.contagion_brd(testGraph, [0], 0.1)
assert len(infected) == 3
infected = hw2_p9.contagion_brd(testGraph, [0], 0.501)
assert len(infected) == 1
infected = hw2_p9.contagion_brd(testGraph, [0], 0.7)
assert len(infected) == 1

testFbGraph = hw2_p9.create_fb_graph("facebook_combined.txt")
assert testFbGraph.number_of_nodes() == 4039
assert testFbGraph.check_edge(107, 1453) == True
infected = hw2_p9.contagion_brd(testFbGraph, [0], 0)
assert len(infected) == 4039
infected = hw2_p9.contagion_brd(testFbGraph, [0], 1.01)
assert len(infected) == 1

print("Problem 9 sanity check passed")

# Problem 10 sanity check
testGraph = hw2_p10.WeightedDirectedGraph(3)
testGraph.set_edge(0, 1, 1)
testGraph.set_edge(1, 2, 2)
assert testGraph.number_of_nodes() == 3
assert testGraph.edges_from(0) == [1]
assert testGraph.edges_from(1) == [2]
assert len(testGraph.edges_from(2)) == 0
assert testGraph.get_edge(0,1) == 1
assert testGraph.get_edge(1,2) == 2
assert testGraph.get_edge(2,1) == 0
assert testGraph.get_edge(1,1) == 0
assert testGraph.get_edge(0,2) == 0

v, F = hw2_p10.max_flow(testGraph, 0, 2)
assert v == 1
assert F.number_of_nodes() == 3
assert F.get_edge(0,1) == 1
assert F.get_edge(1,2) == 1
assert F.get_edge(2,1) == -1
assert F.get_edge(1,0) == -1

# 3 drivers, 2 riders
C = [[0, 0], [1, 0], [0, 1]]
result = hw2_p10.max_matching(3, 2, C)
assert len(result) == 3
assert result == [None, 0, 1]

C = hw2_p10.random_driver_rider_bipartite_graph(2, 1)
assert C == [[1, 1],[1,1]]

print("Problem 10 sanity check passed")

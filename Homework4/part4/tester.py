import Homework4.hw4 as hw4

# This file is provided for your convenience.
# These tests are by no means comprehensive, and are just a sanity check.
# Please write your own tests.

testGraph = hw4.DirectedGraph(5)
i = 0
assert testGraph.number_of_nodes() == 5
print("passed test" , i)
i+=1
assert testGraph.get_edge(0,1) == False
print("passed test" , i)
i+=1
testGraph.add_edge(0,1)

assert testGraph.get_edge(0,1) == True
print("passed test" , i)
i+=1
weights = hw4.scaled_page_rank(testGraph,0)

assert weights[2] == 1/5.0
print("passed test" , i)
i+=1
assert hw4.graph_15_1_left().number_of_nodes() == 4
print("passed test" , i)
i+=1
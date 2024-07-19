import unittest

from hw2_p9 import UndirectedGraph, contagion_brd


class TestUndirectedGraph(unittest.TestCase):
    def setUp(self):
        self.graph = UndirectedGraph(5)

    def test_add_edge(self):
        self.graph.add_edge(0, 1)
        self.assertTrue(self.graph.check_edge(0, 1))
        self.assertTrue(self.graph.check_edge(1, 0))

    def test_add_same_edge_twice(self):
        self.graph.add_edge(0, 1)
        self.graph.add_edge(0, 1)
        self.assertTrue(self.graph.check_edge(0, 1))
        self.assertTrue(self.graph.check_edge(1, 0))
        self.assertEqual(self.graph.edges_from(0), [1])
        self.assertEqual(self.graph.edges_from(1), [0])

    def test_edges_from(self):
        self.graph.add_edge(0, 1)
        self.graph.add_edge(0, 2)
        self.graph.add_edge(0, 3)
        self.assertListEqual(sorted(self.graph.edges_from(0)), [1, 2, 3])
        self.assertListEqual(self.graph.edges_from(4), [])

    def test_check_edge(self):
        self.graph.add_edge(0, 1)
        self.assertTrue(self.graph.check_edge(0, 1))
        self.assertTrue(self.graph.check_edge(1, 0))
        self.assertFalse(self.graph.check_edge(0, 2))

    # Note that we added function number_of_nodes function to class UndirectedGraph
    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)

    # def test_get_nodes(self):
    #     self.graph.add_edge(0, 1)
    #     self.graph.add_edge(2, 3)
    #     self.assertListEqual(sorted(self.graph.get_nodes()), [0, 1, 2, 3])


class TestContagionBRD(unittest.TestCase):
    def setUp(self):
        # Create a cycle graph with 6 nodes
        self.graph = UndirectedGraph(6)
        self.graph.add_edge(0, 1)
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(3, 4)
        self.graph.add_edge(4, 5)
        self.graph.add_edge(5, 0)

    def test_empty_graph(self):
        empty_graph = UndirectedGraph(0)
        result = contagion_brd(empty_graph, [], 0.5)
        self.assertListEqual(result, [])

    def test_no_initial_adopters(self):
        result = contagion_brd(self.graph, [], 0.5)
        self.assertListEqual(result, [])

    def test_threshold_zero(self):
        result = contagion_brd(self.graph, [0], 0.0)
        self.assertListEqual(sorted(result), [0, 1, 2, 3, 4, 5])

    def test_single_initial_adopter(self):
        result = contagion_brd(self.graph, [0], 0.6)
        self.assertListEqual(sorted(result), [0])

    def test_multiple_initial_adopters(self):
        result = contagion_brd(self.graph, [0, 2], 0.5)
        self.assertListEqual(sorted(result), [0, 1, 2, 3, 4, 5])

    def test_high_threshold(self):
        result = contagion_brd(self.graph, [0, 1], 0.75)
        self.assertListEqual(sorted(result), [0, 1])

    def test_complete_contagion(self):
        result = contagion_brd(self.graph, [0, 3], 0.5)
        self.assertListEqual(sorted(result), [0, 1, 2, 3, 4, 5])

    def test_no_contagion(self):
        result = contagion_brd(self.graph, [0], 1.0)
        self.assertListEqual(sorted(result), [0])

    def test_contagion_thresholds_fig_4_1_left(self):
        fig_4_1_left = UndirectedGraph(4)
        fig_4_1_left.add_edge(0, 1)
        fig_4_1_left.add_edge(1, 2)
        fig_4_1_left.add_edge(2, 3)

        early_adopters = [0, 1]

        for threshold in [i * 0.01 for i in range(101)]:
            result = contagion_brd(fig_4_1_left, early_adopters, threshold)
            if threshold <= 0.5:
                self.assertListEqual(sorted(result), [0, 1, 2, 3])
            else:
                self.assertNotEqual(sorted(result), [0, 1, 2, 3])

    def test_contagion_thresholds_fig_4_1_right(self):
        fig_4_1_right = UndirectedGraph(7)
        fig_4_1_right.add_edge(0, 1)
        fig_4_1_right.add_edge(1, 2)
        fig_4_1_right.add_edge(1, 3)
        fig_4_1_right.add_edge(3, 4)
        fig_4_1_right.add_edge(3, 5)
        fig_4_1_right.add_edge(5, 6)

        early_adopters = [0, 1, 2]

        for threshold in [i * 0.01 for i in range(101)]:
            result = contagion_brd(fig_4_1_right, early_adopters, threshold)
            if threshold <= 0.33:
                self.assertListEqual(sorted(result), [0, 1, 2, 3, 4, 5, 6])
            else:
                self.assertNotEqual(sorted(result), [0, 1, 2, 3, 4, 5, 6])

    def test_disconnected_graph(self):
        disconnected_graph = UndirectedGraph(2)
        early_adopters = [0]
        for threshold in [i * 0.01 for i in range(101)]:
            result = contagion_brd(disconnected_graph, early_adopters, threshold)
            self.assertEqual(sorted(result), [0])


if __name__ == '__main__':
    unittest.main()

from preserve_bounds import preserve_bounds
import unittest
import numpy as np
from numpy.testing import *

from contract_pair import update_positions, update_Q, update_pairs, update_face, update_features, sort_by_error

class ContractPairTests(unittest.TestCase):

    def test_update_positions(self):
        pair = np.array([1., 0., 1., 9., 9., 9.])

        positions = np.array([
                [0., 0., 0.],
                [1., 1., 1.],
            ])

        solution = np.array([
                [9., 9., 9.]
            ])

        res = update_positions(pair, positions)

        assert_array_equal(res, solution)

    def test_update_Q(self):
        pair = np.array([1., 0., 1., 9., 9., 9.])

        Q = np.array([
                [[1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.]],
                [[1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.]],
            ])

        solution = np.array([
                [[2., 2., 2., 2.],
                 [2., 2., 2., 2.],
                 [2., 2., 2., 2.],
                 [2., 2., 2., 2.]],
            ])

        res = update_Q(pair, Q)

        assert_array_equal(res, solution)

    def test_update_pairs(self):
        pairs = np.array([
                [1., 0., 1., 9., 9., 9.],
                [1., 0., 1., 8., 9., 9.],
                [5., 1., 2., 9., 8., 9.],
                [3., 2., 3., 9., 9., 8.],
            ])

        positions = np.array([
                [0., 0., 0.],
                [1., 1., 1.],
            ])

        Q = np.array([
                [[1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.]],
                [[1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.],
                 [1., 1., 1., 1.]],
            ])

        solution = np.array([
                [2., 0., 0., 0., 0., 0.], # invalid pairs are removed in simplify.pyx
                [2., 0., 1., 0., 0., 0.],
                [3., 1., 2., 9., 9., 8.]
            ])

        res = update_pairs(pairs, positions, Q, None)

        assert_array_equal(res, solution)

    def test_update_face(self):
        pair = np.array([1., 1., 3., 9., 9., 9.])

        positions = np.array([
                [0., 0., 0.],
                [1., 1., 1.],
                [2., 1., 0.],
                [-0., -1., 0.],
                [-1., -1., -1.],
                [-2., -1., 0.],
            ])

        face = np.array([
                [0, 1, 2],
                [5, 1, 3],
                [1, 3, 4],
                [5, 3, 4]
            ])

        # all faces with 1, 3 are removed
        # all ids > 3 are diminished by 1
        # duplicates are removed
        solution = np.array([
                [0, 1, 2],
                [4, 1, 3]
            ])

        res = update_face(pair, face)

        assert_array_equal(res, solution)

    def test_update_features(self):
        pair = np.array([1., 1., 3., 9., 9., 9., -1, -2, -3])

        positions = np.array([
                [0., 0., 0.],
                [1., 1., 1.],
                [2., 1., 0.],
                [-0., -1., 0.],
                [-1., -1., -1.],
                [-2., -1., 0.],
            ])

        features = np.array([
                [0, 1, 2],
                [5, 1, 3],
                [1, 3, 4],
                [5, 3, 4]
            ])

        solution = np.array([
                [0, 1, 2],
                [-1, -2, -3],
                [1, 3, 4]
            ])

        res = update_features(pair, features)

        assert_array_equal(res, solution)

        res = update_features(pair, None)

        assert_array_equal(res, None)

    def test_sort_by_error(self):
        pairs = np.array([
                [1., 0., 1., 9., 9., 9.],
                [1., 0., 1., 9., 9., 9.],
                [1., 0., 1., 8., 9., 9.],
                [5., 1., 2., 9., 8., 9.],
                [3., 2., 3., 9., 9., 8.],
            ])

        # sorts only by the first column (if tie also by the rest)
        # removes duplicates
        solution = np.array([
                [1., 0., 1., 8., 9., 9.],
                [1., 0., 1., 9., 9., 9.],
                [3., 2., 3., 9., 9., 8.],
                [5., 1., 2., 9., 8., 9.],
            ])

        res = sort_by_error(pairs)

        assert_array_equal(res, solution)


def np_not_equal(arr1, arr2):
    assert_raises(AssertionError, assert_array_equal, arr1, arr2)

if __name__ == '__main__':
    unittest.main()
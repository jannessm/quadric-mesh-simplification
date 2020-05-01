from preserve_bounds import preserve_bounds
import unittest
import numpy as np
from numpy.testing import *

from heap import PairHeap
from contract_pair import update_pairs, update_face, update_features
import array

class ContractPairTests(unittest.TestCase):

    def test_update_pairs(self):
        pairs = np.array([
               [1., 0., 1., 9., 9., 9.],
               [1., 0., 1., 8., 9., 9.],
               [5., 1., 2., 9., 8., 9.],
               [-4., 2., 3., 9., 9., 8.],
               [3., 2., 3., 9., 9., 8.],
           ])

        positions = np.array([
               [0., 0., 0.],
               [1., 1., 1.],
               [-1., 0., 1.],
               [0., 1., -1.],
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
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],
               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]
           ])

        solution = np.array([
            [-10e6, 0., 1., 9., 9., 9.], # invalid pairs are removed in simplify.pyx
            [-10e6, 0., 1., 8., 9., 9.],
            [2., 0., 2., -1., 0., 1.],
            [-4., 2., 3., 9., 9., 8.], # is not completly sorted because its a heap
            [3., 2., 3., 9., 9., 8.],
           ])

        heap = PairHeap(pairs)

        update_pairs(0, 1, heap, positions, Q, None)

        self.assertEqual(heap.length(), solution.shape[0])

        for i in range(solution.shape[0]):
            np.testing.assert_almost_equal(heap.get_pair(i), solution[i])

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

        deleted_faces = array.array('B', [False, False, False, False])

        # all faces with 1, 3 are removed
        # all ids > 3 are diminished by 1
        # duplicates are removed
        solution = array.array('I', [False, True, True, False])

        update_face(pair[1], pair[2], face, deleted_faces)

        self.assertEqual(deleted_faces, solution)

    def test_update_features(self):
        pair = np.array([1., 1., 3., 9., 9., 9., -1, -2, -3])

        features = np.array([
                [0, 1, 2],
                [5, 1, 3],
                [1, 3, 4],
                [5, 3, 4]
            ], dtype=np.double)

        solution = np.array([
                [0, 1, 2],  # features are removed at the end
                [-1, -2, -3],
                [1, 3, 4],
                [5, 3, 4]
            ])

        update_features(pair, features)

        assert_array_equal(features, solution)

        update_features(pair, None) # throws no error


def np_not_equal(arr1, arr2):
    assert_raises(AssertionError, assert_array_equal, arr1, arr2)

if __name__ == '__main__':
    unittest.main()
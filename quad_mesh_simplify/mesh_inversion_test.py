from mesh_inversion import has_mesh_inversion
import unittest
import numpy as np
from numpy.testing import *

import os.path as osp

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)),'..'))

from testing_utils import plot_test_mesh

import array

DEBUG = False

class MeshInversionTests(unittest.TestCase):

    def test_non_inverted_mesh(self):
        positions = np.array([
            [0.25, 0, 0.],
            [-.25, 0, 0.],
            [0.5, .5, 1.],
            [-.5, 0.5, 1.],
            [0.75, 0., 0.],
        ])

        new_pos = np.array([
            [0.25, 0, 0.],
            [0.5, .5, 1.],
            [-.5, 0.5, 1.],
            [0.75, 0., 0.],
        ])

        face = np.array([
            [0,2,3],
            [0,1,3],
            [0,4,2],
        ])

        new_face = np.array([
            [0,1,2],
            [0,3,1],
        ])

        deleted_faces = array.array('B', [False, False, False])

        if DEBUG:
            plot_test_mesh(positions, face)
            plot_test_mesh(new_pos, new_face)

        self.assertFalse(has_mesh_inversion(0, 1, positions, new_pos, face, deleted_faces))

    def test_inverted_mesh(self):
        positions = np.array([
            [0.25, 0, 0.],
            [-.25, 0, 0.],
            [0.5, .5, 1.],
            [-.5, 0.5, 1.],
            [0.75, 0., 0.],
        ])

        new_pos = np.array([
            [0.25, 0, 0.],
            [-.25, 0, 0.],
            [0.5, .5, 1.],
            [-.5, 0.5, 1.],
        ])

        face = np.array([
            [0,2,3],
            [0,1,3],
            [0,4,2],
        ])

        new_face = np.array([
            [0,2,3],
            [0,1,3],
            [0,1,2],
        ])

        if DEBUG:
            plot_test_mesh(positions, face)
            plot_test_mesh(new_pos, new_face)

        deleted_faces = array.array('B', [False, False, False])

        self.assertTrue(has_mesh_inversion(1, 4, positions, new_pos, face, deleted_faces))

if __name__ == '__main__':
    unittest.main()
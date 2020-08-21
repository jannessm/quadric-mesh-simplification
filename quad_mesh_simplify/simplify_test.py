import os.path as osp
import sys
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__)),'..'))

from quad_mesh_simplify import simplify_mesh
import unittest
import numpy as np

from testing_utils import plot_test_mesh

DEBUG = True

class SimplifyTests(unittest.TestCase):

    def test_wrong_params(self):
        pos = np.array([
            [-1., -1., -1.],
            [-.5, 0., 0.],
            [-1., 1., 1.],
            [0., 0.25, 0.25],
            [0., -0.25, -0.25],
            [1., -1., -1.],
            [.5, 0., 0.],
            [1., 1., 1.],
            [0., -1., -1.],
            [0., 1., 1.],
        ])
        face = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [1, 2, 3],
            [3, 6, 7],
            [3, 4, 6],
            [4, 5, 6],
            [0, 8, 4],
            [5, 4, 8],
            [2, 3, 9],
            [3, 9, 7],
            [5, 6, 7],
            [0, 1, 2]
        ])
        
        try:
            simplify_mesh("hi", face, 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, 'hi', 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(np.zeros((1,5)), face, 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(np.zeros((1,3), np.int), face, 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, 'hi', 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, np.zeros((1,5)), 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, np.zeros((1,3), np.double), 5)
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, face, 'hi')
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, face, 100, features='hi')
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, face, 100, features=np.zeros((3,3)))
            fail()
        except:
            pass

        try:
            simplify_mesh(pos, face, 100, features=np.zeros((pos.shape[0], 3), np.int))
            fail()
        except:
            pass

    def test_simplify_mesh_without_threshold(self):
        pos = np.array([
            [-1., -1., -1.],
            [-.5, 0., 0.],
            [-1., 1., 1.],
            [0., 0.25, 0.25],
            [0., -0.25, -0.25],
            [1., -1., -1.],
            [.5, 0., 0.],
            [1., 1., 1.],
            [0., -1., -1.],
            [0., 1., 1.],
        ], np.double)
        face = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [1, 2, 3],
            [3, 6, 7],
            [3, 4, 6],
            [4, 5, 6],
            [0, 8, 4],
            [5, 4, 8],
            [2, 3, 9],
            [3, 9, 7],
            [5, 6, 7],
            [0, 1, 2]
        ], np.uint32)

        if DEBUG:
            plot_test_mesh(pos, face, False)

        for i in range(1, 8):
            res_pos, res_face = simplify_mesh(np.copy(pos), np.copy(face), 10 - i)
            print(res_pos)
            print(res_face)
            
            if DEBUG:
                plot_test_mesh(res_pos, res_face)
            self.assertEqual(res_pos.shape, (10 - i, 3))        

    def test_simplify_mesh_with_threshold(self):
        pos = np.array([
           [-2., -2., -2.],
           [-2., 0., 0.],
           [0., 0.25, 0.],
           [2., -2., -2.],
           [2., 0., 0.],
           [0., -0.25, 0.],
        ])
        face = np.array([
           [0, 1, 2],
           [5, 3, 4]
        ], np.uint32)

        new_pos = np.array([
           [-2., -2., -2.],
           [-2., 0., 0.],
           [0., 0., 0.],
           [2., -2., -2.],
           [2., 0., 0.],
        ])
        new_face = np.array([
           [0, 1, 2],
           [2, 3, 4]
        ])

        if DEBUG:
            plot_test_mesh(pos, face, False)

        res_pos, res_face = simplify_mesh(np.copy(pos), np.copy(face), 5, threshold=0.6)

        if DEBUG:
            plot_test_mesh(res_pos, res_face)

        np.testing.assert_equal(res_face, new_face)
        np.testing.assert_equal(res_pos, new_pos)

if __name__ == '__main__':
    unittest.main()
from simplify import simplify_mesh
import unittest
import numpy as np

from testing_utils import plot_test_mesh

class SimplifyTests(unittest.TestCase):

	def test_simplify_mesh_without_threshold(self):
		pos = np.array([
		    [-1., -1., -1.],
		    [-1., 0., 0.],
		    [-1., 1., 1.],
		    [0., 0.25, 0.25],
		    [0., -0.25, -0.25],
		    [1., -1., -1.],
		    [1., 0., 0.],
		    [1., 1., 1.],
		    [0., -1., -1.],
		    [0., 1., 1.],
		]) - 2
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
		    [3, 9, 7]
		])

		new_pos = np.array([
		    [-1., -1., -1.],
		    [-1., 0., 0.],
		    [-1., 1., 1.],
		    [0., 0., 0.],
		    [1., -1., -1.],
		    [1., 0., 0.],
		    [1., 1., 1.],
		])
		new_face = np.array([
		    [0, 1, 3],
		    [1, 2, 3],
		    [3, 5, 6],
		    [3, 4, 5]
		])

		res_pos, res_face = simplify_mesh(pos, face, 8)
		print(res_pos)
		print(res_face)
		plot_test_mesh(res_pos, res_face)

		#np.testing.assert_equal(res_pos, new_pos)
		#np.testing.assert_equal(res_face, new_face)

if __name__ == '__main__':
	unittest.main()
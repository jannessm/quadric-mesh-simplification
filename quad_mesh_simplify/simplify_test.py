from simplify import simplify_mesh
import unittest
import numpy as np

class SimplifyTests(unittest.TestCase):

	def test_simplify_mesh_without_threshold(self):
		positions = np.array([
		    [0., 0., 0.],
		    [1., 0., 0.],
		    [2., 2., 0.],
		    [2., 1., 1.],
		    [0., 0., 0.5],
		])

		face = np.array([
		    [0, 1, 2],
		    [0, 2, 3],
		])

		solution_positions = []
		solution_face = []

		res_pos, res_face = simplify_mesh(positions, face, 3)
		
		print(res_pos)
		print(res_face)

if __name__ == '__main__':
	unittest.main()
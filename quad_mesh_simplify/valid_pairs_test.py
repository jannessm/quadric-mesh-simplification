from valid_pairs import compute_valid_pairs
import unittest
import numpy as np

class ValidPairsTests(unittest.TestCase):

	def test_valid_edges(self):
		positions = np.array([
			[0., 0., 0.],
			[1., 0., 0.],
			[1., 1., 0.],
			[1., 1., 1.],
			[1., 0., 1.],
		])

		face = np.array([
			[0, 1, 2],
			[1, 2, 3],
		])

		pairs = compute_valid_pairs(positions, face, 0)

		res = np.array([
			[0, 1],
			[0, 2],
			[1, 2],
			[1, 3],
			[2, 3],
		])

		np.testing.assert_equal(pairs, res)

	def test_valid_pairs(self):
		positions = np.array([
			[0., 0., 0.],
			[1., 0., 0.],
			[1., 1., 0.],
			[1., 1., 1.],
			[1., 0., 1.],
		])

		face = np.array([
			[0, 1, 2],
			[1, 2, 3],
		])

		pairs = compute_valid_pairs(positions, face, 2)

		res = np.array([
			[0, 1],
			[0, 2],
			[0, 3],
			[0, 4],
			[1, 2],
			[1, 3],
			[1, 4],
			[2, 3],
			[2, 4],
			[3, 4],
		])

		np.testing.assert_equal(pairs, res)

if __name__ == '__main__':
	unittest.main()
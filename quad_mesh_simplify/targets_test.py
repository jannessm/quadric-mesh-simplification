from targets import compute_targets
from pair import Pair
import unittest
import numpy as np

class TargetsTests(unittest.TestCase):

	def test_compute_targets(self):
		positions = np.array([
			[0., 0., 0.],
			[1., 0., 0.],
			[1., 1., 0.],
			[1., 1., 1.],
			[1., 0., 1.],
		])

		Q = np.array([
			[[2., 2., 2., 2.],
			 [2., 2., 2., 2.],
			 [2., 2., 2., 2.],
			 [2., 2., 2., 2.]],
			[[1., 1., 1., 1.],
			 [1., 1., 1., 1.],
			 [1., 1., 1., 1.],
			 [1., 1., 1., 1.]],
			[[2., 2., 2., 2.],
			 [2., 2., 2., 2.],
			 [2., 2., 2., 2.],
			 [2., 2., 2., 2.]],
			[[1., 1., 1., 1.],
			 [1., 1., 1., 1.],
			 [1., 1., 1., 1.],
			 [1., 1., 1., 1.]],
		])

		valid_pairs = np.array([
			[0, 1],
			[1, 2],
		])

		solution = [
			Pair(),
			Pair()
		]
		solution[0].calculate_error(0, 1, positions, Q, None)
		solution[1].calculate_error(1, 2, positions, Q, None)

		res = compute_targets(positions, Q, valid_pairs, None)
		
		for i, r in enumerate(res):
			self.assertEqual(r, solution[i])

if __name__ == '__main__':
	unittest.main()
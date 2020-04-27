from targets import compute_targets, calculate_pair_attributes
from pair import Pair
import unittest
import numpy as np

class TargetsTests(unittest.TestCase):

	def test_calculate_pair_attributes(self):
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

		solution_err = 3.
		solution_target = [0., 0., 0.]
		p = calculate_pair_attributes(0, 1, positions, Q, None)
		self.assertEqual(p[0], solution_err)
		self.assertEqual(p[1], 0) #v1
		self.assertEqual(p[2], 1) #v2
		np.testing.assert_equal(p[3:], solution_target)

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
			Pair().calculate_error(0, 1, positions, Q, None),
			Pair().calculate_error(1, 2, positions, Q, None)
		]

		res = compute_targets(positions, Q, valid_pairs, None)
		
		for i, r in enumerate(res):
			self.assertEqual(r, solution[i])

if __name__ == '__main__':
	unittest.main()
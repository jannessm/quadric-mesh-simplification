from pair import Pair
import unittest
import numpy as np

class PairTests(unittest.TestCase):

	def test_calculate_error(self):
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
		p = Pair().calculate_error(0, 1, positions, Q, None)
		self.assertEqual(p.error, solution_err)
		np.testing.assert_equal(p.target, solution_target)

if __name__ == '__main__':
	unittest.main()
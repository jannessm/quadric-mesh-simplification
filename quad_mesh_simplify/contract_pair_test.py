from contract_pair import contract_first_pair
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

		pairs = [
			Pair().calculate_error(0, 1, positions, Q, None),
			Pair().calculate_error(1, 2, positions, Q, None)
		]

		res_pairs, res_positions, res_Q, res_features = contract_first_pair(pairs, positions, Q, None)
		print(res_pairs, res_positions, res_Q, res_features)

if __name__ == '__main__':
	unittest.main()
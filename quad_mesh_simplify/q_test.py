import unittest
import numpy as np

from q import compute_Q

class TestComputeQ(unittest.TestCase):
	
	def test_compute_Q(self):
		positions = np.array([
			[0., 0., 0.],
			[1., 0., 0.],
			[1., 1., 0.],
			[1., 1., 1.],
		])

		face = np.array([
			[0, 1, 2],
			[0, 2, 3],
		])

		res = np.array([
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
		Q = compute_Q(positions, face)
		np.testing.assert_almost_equal(Q, res)

if __name__ == '__main__':
	unittest.main()

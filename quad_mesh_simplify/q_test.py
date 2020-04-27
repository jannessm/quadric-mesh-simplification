import unittest
import numpy as np

from q import compute_Q

tolerance = 10e-3

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
			[[2., 2., 2., 2.]
			 [2., 2., 2., 2.]
			 [2., 2., 2., 2.]
			 [2., 2., 2., 2.]],
			[[1., 1., 1., 1.]
			 [1., 1., 1., 1.]
			 [1., 1., 1., 1.]
			 [1., 1., 1., 1.]],
			[[2., 2., 2., 2.]
			 [2., 2., 2., 2.]
			 [2., 2., 2., 2.]
			 [2., 2., 2., 2.]],
			[[1., 1., 1., 1.]
			 [1., 1., 1., 1.]
			 [1., 1., 1., 1.]
			 [1., 1., 1., 1.]]
		])

		self.assertLess(compute_Q - res, tolerance)

if __name__ == '__main__':
	unittest.main()

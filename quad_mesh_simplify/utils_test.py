from utils import get_faces_for_node
import unittest
import numpy as np

class UtilsTests(unittest.TestCase):

	def test_get_faces_for_node(self):
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

		np.testing.assert_equal(get_faces_for_node(0, face), np.array([[0, 1, 2]]))

		np.testing.assert_equal(get_faces_for_node(4, face), np.zeros((0,3)))

if __name__ == '__main__':
	unittest.main()
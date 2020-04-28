from preserve_bounds import preserve_bounds
import unittest
import numpy as np
from numpy.testing import *

import os.path as osp

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)),'..'))

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
		])
		
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

		Q = np.array([
			   [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]],

		       [[3., 3., 3., 3.],
		        [3., 3., 3., 3.],
		        [3., 3., 3., 3.],
		        [3., 3., 3., 3.]],

		       [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]],

		       [[6., 6., 6., 6.],
		        [6., 6., 6., 6.],
		        [6., 6., 6., 6.],
		        [6., 6., 6., 6.]],

		       [[6., 6., 6., 6.],
		        [6., 6., 6., 6.],
		        [6., 6., 6., 6.],
		        [6., 6., 6., 6.]],

		       [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]],

		       [[3., 3., 3., 3.],
		        [3., 3., 3., 3.],
		        [3., 3., 3., 3.],
		        [3., 3., 3., 3.]],

		       [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]],

		       [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]],

		       [[2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.],
		        [2., 2., 2., 2.]]
		    ])


		new_Q = preserve_bounds(pos, face, np.copy(Q))

		# inner nodes were not be affected by penalty
		assert_equal(Q[3:5], new_Q[3:5])
		# outers yes
		np_not_equal(Q[:3], new_Q[:3])
		np_not_equal(Q[5:], new_Q[5:])

def np_not_equal(arr1, arr2):
	assert_raises(AssertionError, assert_array_equal, arr1, arr2)

if __name__ == '__main__':
	unittest.main()
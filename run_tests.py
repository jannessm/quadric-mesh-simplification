import unittest
import os.path as osp

if __name__ == '__main__':
	loader = unittest.TestLoader()
	suite = loader.discover(
		osp.join(osp.abspath(osp.dirname(__file__)), 'quad_mesh_simplify'),
		pattern='*_test.py'
	)
	unittest.TextTestRunner(verbosity=2).run(suite)
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import os.path as osp

__version__ = '0.0.1'

url = 'https://github.com/jannessm/quadric-mesh-simplification'

files = [
	'simplify.c',
	'array.c',
	'clean_mesh.c',
	'contract_pair.c',
	'edges.c',
	'maths.c',
	'mesh_inversion.c',
	'pair_heap.c',
	'pair.c',
	'preserve_bounds.c',
	'q.c',
	'targets.c',
	'upper_tri.c',
	'valid_pairs.c',
	'test_utils.c'
]

ext_modules = [
	Extension(
		'simplify',
		[osp.join(osp.dirname(osp.abspath(__file__)),'quad_mesh_simplify', 'c', f) for f in files] + [osp.join(osp.dirname(osp.abspath(__file__)),'quad_mesh_simplify','simplify.pyx')],
		# extra_compile_args=['-fopenmp'],
		# extra_link_args=['-fopenmp'],
		include_dirs=[np.get_include()],
		define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_17_API_VERSION")],
	),
]

ext_modules = cythonize(ext_modules)

setup(
  name='quad_mesh_simplify',
  version=__version__,
  author='Jannes Magnusson',
  url=url,
  ext_modules=ext_modules,
  zip_safe=False
)
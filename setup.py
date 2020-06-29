from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

FROM_SOURCE = True

__version__ = '0.0.1'

url = 'https://github.com/jannessm/quadric-mesh-simplification'

ext = '.pyx' if FROM_SOURCE else '.c'

files = [
	'quad_mesh_simplify.clean_mesh',
	'quad_mesh_simplify.contract_pair',
	'quad_mesh_simplify.maths',
	'quad_mesh_simplify.heap',
	'quad_mesh_simplify.mesh_inversion',
	'quad_mesh_simplify.preserve_bounds',
	'quad_mesh_simplify.q',
	'quad_mesh_simplify.simplify',
	'quad_mesh_simplify.targets',
	'quad_mesh_simplify.valid_pairs',
]

files = [(f, f.replace('.', '/') + ext) for f in files]

ext_modules = [
	Extension(
		f[0],
		[f[1]],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_17_API_VERSION")],
	)
	for f in files
]

if FROM_SOURCE:
    ext_modules = cythonize(ext_modules, annotate=True)

setup(
  name='quad_mesh_simplify',
  version=__version__,
  author='Jannes Magnusson',
  url=url,
  ext_modules=ext_modules,
  zip_safe=False
)
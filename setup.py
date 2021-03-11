from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import os.path as osp

__version__ = '1.1.4'

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

src_path = osp.join(osp.dirname(osp.abspath(__file__)), 'quad_mesh_simplify')

ext_modules = [
	Extension(
		'simplify',
		[osp.join(src_path, 'c', f) for f in files] + [osp.join(src_path,'simplify.pyx')],
		# extra_compile_args=['-fopenmp'],
		# extra_link_args=['-fopenmp'],
		include_dirs=[np.get_include()],
		define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_17_API_VERSION")],
	),
]

ext_modules = cythonize(ext_modules)

with open("README.md", "r") as fh:
	long_description = fh.read()

def parse_requirements(filename):
	"""Load requirements from a pip requirements file."""
	lineiter = (line.strip() for line in open(filename))
	return [line for line in lineiter if line and not line.startswith("#")]

setup(
  name='quad_mesh_simplify',
  version=__version__,
  author='Jannes Magnusson',
  url=url,
	description="Simplify meshes including vertex features.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	install_requires=parse_requirements("requirements.txt"),
  python_requires=">=3.6.3",
  ext_modules=ext_modules,
  zip_safe=False,
)
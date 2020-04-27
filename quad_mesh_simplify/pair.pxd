import numpy as np
cimport numpy as np

np.import_array()

cdef class Pair:
	cdef double error
	cdef long v1, v2
	cdef np.ndarray target, new_features
	
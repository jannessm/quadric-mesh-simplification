import numpy as np
cimport numpy as np

cdef class Pair:
	cdef public double error
	cdef public long v1, v2
	cdef public np.ndarray target, new_features

	cpdef Pair calculate_error(Pair, long, long, np.ndarray, np.ndarray, np.ndarray)
	
	cdef np.ndarray make_homogeneous(Pair, np.ndarray)

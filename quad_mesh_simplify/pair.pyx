import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

cdef class Pair:
	r"""managing a valid pair by containing the ids of the nodes and the new contracted node position"""
	
	cdef public int id

	cdef public long v1
	cdef public long v2

	cdef public np.ndarray new_v

	def __init__(self):
		pass
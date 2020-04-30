cimport numpy as np

ctypedef np.long_t DTYPE_LONG_T
ctypedef np.double_t DTYPE_DOUBLE_T

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] compute_Q(
	double [:, :] positions,
	long [:, :] face)
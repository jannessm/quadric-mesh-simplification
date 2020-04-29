cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef void preserve_bounds(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q)

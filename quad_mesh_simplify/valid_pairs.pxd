cimport numpy as np

ctypedef np.int64_t DTYPE_LONG_T
ctypedef np.double_t DTYPE_DOUBLE_T

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] compute_valid_pairs(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    double threshold)
cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] compute_targets(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q,
    np.ndarray[DTYPE_LONG_T, ndim=2] valid_pairs,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features)

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] calculate_pair_attributes(
    long v1,
    long v2,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features)

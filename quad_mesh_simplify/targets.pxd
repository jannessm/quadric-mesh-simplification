cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] compute_targets(
    double [:, :] positions,
    double [:, :, :] Q,
    long [:, :] valid_pairs,
    double [:, :] features)

cpdef void calculate_pair_attributes(
    long v1,
    long v2,
    double [:, :] positions,
    double [:, :, :] Q,
    double [:, :] features,
    double [:] pair)

cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef int has_mesh_inversion(
    long v1,
    long v2,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] new_positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face)
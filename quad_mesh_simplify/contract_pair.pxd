cimport numpy as np

cdef int target_offset 
cdef int feature_offset

ctypedef np.long_t DTYPE_LONG_T
ctypedef np.double_t DTYPE_DOUBLE_T

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] update_positions(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions)

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] update_Q(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q)

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] update_pairs(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features)

cpdef void update_face(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    list deleted_faces)

cpdef void update_features(
    np.ndarray[DTYPE_DOUBLE_T, ndim=1] pair,
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features)

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] sort_by_error(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pairs)
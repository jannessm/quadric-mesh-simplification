from cpython cimport array
cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] clean_positions(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pos,
    array.array deleted_pos)

cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] clean_features(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] features,
    array.array deleted_pos)

cdef np.ndarray[DTYPE_LONG_T, ndim=2] clean_face(
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    array.array deleted_faces,
    char [:] deleted_pos)
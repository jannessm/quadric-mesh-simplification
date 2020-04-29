cimport numpy as np

ctypedef np.long_t DTYPE_LONG_T
ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.uint8_t DTYPE_BOOL_T

cpdef np.ndarray[DTYPE_LONG_T, ndim=2] get_faces_for_node(
    long node_id,
    np.ndarray[DTYPE_LONG_T, ndim=2] face)

cpdef np.ndarray[DTYPE_LONG_T, ndim=1] get_rows(
    np.ndarray[DTYPE_BOOL_T] condition)

cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] face_normal(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] pos,
    DTYPE_BOOL_T normalized,
    int reference_id)
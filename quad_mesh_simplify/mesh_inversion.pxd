cimport numpy as np

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef int has_mesh_inversion(
    long v1,
    long v2,
    double [:, :] positions,
    double [:, :] new_positions,
    long [:, :] face,
    char [:] deleted_faces)
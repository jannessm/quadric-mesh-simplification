cimport numpy as np

cdef int target_offset 
cdef int feature_offset

ctypedef np.long_t DTYPE_LONG_T
ctypedef np.double_t DTYPE_DOUBLE_T

from cpython cimport array
from .heap cimport PairHeap

cpdef void update_pairs(
    long v1,
    long v2,
    PairHeap heap,
    double [:, :] positions,
    double [:, :, :] Q,
    double [:, :] features)

cpdef void update_face(
    long v1,
    long v2,
    long [:, :] face,
    char [:] deleted_faces)

cpdef void update_features(
    double [:] pair,
    double [:, :] features)

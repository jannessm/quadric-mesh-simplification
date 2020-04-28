cimport numpy as np

cdef int target_offset 
cdef int feature_offset

cpdef np.ndarray update_positions(np.ndarray, np.ndarray)

cpdef np.ndarray update_Q(np.ndarray, np.ndarray)

cpdef np.ndarray update_pairs(np.ndarray pairs, np.ndarray, np.ndarray, np.ndarray)

cpdef np.ndarray update_face(np.ndarray, np.ndarray)

cpdef np.ndarray update_features(np.ndarray, np.ndarray)

cpdef np.ndarray sort_by_error(np.ndarray)
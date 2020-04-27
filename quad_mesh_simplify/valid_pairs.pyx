import numpy as np
cimport numpy as np

DTYPE_LONG = np.int64

ctypedef np.int64_t DTYPE_LONG_T

from utils cimport get_faces_for_node

cpdef np.ndarray compute_valid_pairs(np.ndarray positions, np.ndarray face, double threshold):
	cdef np.ndarray valid_pairs = np.zeros((0,2), dtype=DTYPE_LONG_T)

	# option 1 to be valid
	cdef np.ndarray f

	for f in face:
		cdef list edges = np.array([
			[f[0], f[1]],
			[f[1], f[2]],
			[f[2], f[0]]
		], dtype=DTYPE_LONG)

		valid_pairs = np.vstack([valid_pairs, edges])
		valid_pairs = remove_duplicates(valid_pairs)

	# option 2: distance below threshold
	cdef int i

	#for i in range(positions.shape[0]):

	return valid_pairs

cdef np.ndarray remove_duplicates(np.ndarray arr):
	return np.unique(np.sort(arr, axis=0), axis=0)
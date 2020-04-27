from pair import Pair
import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

cpdef list compute_targets(
	np.ndarray positions,
	np.ndarray Q,
	np.ndarray valid_pairs,
	np.ndarray features
):
	cdef list pairs = []
	cdef np.ndarray pair, target, Q1, Q2, new_Q, v1, v2, v12, errors, ranges, feature
	cdef double error
	cdef int i

	ranges = np.arange(0, 1.1, 0.1)

	for pair in valid_pairs:
		p = Pair().calculate_error(
			pair[0], pair[1], positions, Q, features
		)
		pairs.append(p)

	return pairs
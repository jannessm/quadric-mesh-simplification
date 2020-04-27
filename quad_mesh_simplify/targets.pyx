from pair import Pair
import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

cpdef np.ndarray compute_targets(
	np.ndarray positions,
	np.ndarray Q,
	np.ndarray valid_pairs,
	np.ndarray features
):
	if features is None:
		features = np.zeros((0,0))

	assert(len(features.shape) == 2)
	
	# a pair consists of error, v1, v2, target, feature
	cdef int pair_shape = 3 + 3 + features.shape[1]
	cdef int target_offset = 3

	cdef np.ndarray pairs = np.zeros((0, pair_shape), dtype=DTYPE_DOUBLE)
	cdef np.ndarray p

	ranges = np.arange(0, 1.1, 0.1)

	for pair in valid_pairs:
		p = calculate_pair_attributes(pair[0], pair[1], positions, Q, features)
		pairs = pairs.vstack([pairs, p])

	return pairs

cpdef np.ndarry calculate_pair_attributes(long v1, long v2, np.ndarray positions, np.ndarray Q, np.ndarray features):
	cdef int pair_shape = 3 + 3 + features.shape[1]
	cdef int target_offset = 3
	cdef int feature_offset = 3 + 3

	cdef np.ndarray pair = np.zeros((pair_shape), dtype=DTYPE_DOUBLE)
	cdef np.ndarray new_Q, errors, p1, p2, p12, ranges
	cdef double error, i
	cdef int min_id

	p[1] = v1
	p[2] = v2

	new_Q = Q[v1] + Q[v2]
	
	# do not use explicit solution because of feature trade-off
	errors = np.zeros((0), dtype=DTYPE_DOUBLE)
	p1 = make_homogeneous(positions[v1])
	p2 = make_homogeneous(positions[v2])

	# calculate errors for a 10 different targets on p1 -> p2
	r = np.arange(0, 1.1, 0.1)[:, None]
	p12 = r.dot(p1[:, None].T) + (1 - r).dot(p2[:, None].T)
	errors = p12.dot(new_Q).dot(p12.T) * np.eye(11)
	errors = np.max(errors, axis=1)

	# get minimal error
	min_id = np.argmin(errors)
	p[0] = errors[min_id]
	i = min_id * 0.1
	p[target_offset: feature_offset] = (i * p1 + (1 - i) * p2)[:3]

	if features is not None:
		p[feature_offset:] = i * features[v1] + (1 - i) * features[v2]

	return pair

cdef np.ndarray make_homogeneous(np.ndarray arr):
		return np.hstack([arr, 1])
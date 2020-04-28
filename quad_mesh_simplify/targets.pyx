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
	"""Computes the optimal position and the resulting feature vector (if provided) of the contracted node for nodes in valid_pairs.
	Since the feature vector should be aggregated, the exact mixture of v1 and v2 are needed and sampled from 10 different positions
	on the edge v1, v2.

	Args:
	    positions (:class:`ndarray`): array of shape (num_nodes, 3) containing the x, y, z position for each node
	    Q (:class:`ndarray`): Q matrixes for each node (shape (num_nodes, 4, 4))
	    valid_pairs (:class:`ndarray`): list of valid_pairs. A pair consists of the error, v1, v2, target position(, feature vector).
	    features (:class:`ndarray`, optional): feature matrix of shape (num_nodes, num_features)

	:rtype: :class:`ndarray`
	"""
	if features is None:
		features = np.zeros((0,0))
	
	# a pair consists of error, v1, v2, target, feature
	cdef int pair_shape = 3 + 3 + features.shape[1]
	cdef int target_offset = 3

	cdef np.ndarray pairs = np.zeros((0, pair_shape), dtype=DTYPE_DOUBLE)
	cdef np.ndarray p

	ranges = np.arange(0, 1.1, 0.1)

	for pair in valid_pairs:
		p = calculate_pair_attributes(pair[0], pair[1], positions, Q, features)
		pairs = np.vstack([pairs, p])

	return pairs

cpdef np.ndarray calculate_pair_attributes(long v1, long v2, np.ndarray positions, np.ndarray Q, np.ndarray features):
	"""Computes the optimal position of the contracted node of the edge (v1, v2).

	Args:
		v1 (:class:`long`): id for first node
		v2 (:class:`long`): id for second node
	    positions (:class:`ndarray`): array of shape (num_nodes, 3) containing the x, y, z position for each node
	    Q (:class:`ndarray`): Q matrixes for each node (shape (num_nodes, 4, 4))
	    features (:class:`ndarray`, optional): feature matrix of shape (num_nodes, num_features)

	:rtype: :class:`ndarray`
	"""
	if features is None:
		features = np.zeros((0,0))

	cdef int pair_shape = 3 + 3 + features.shape[1]
	cdef int target_offset = 3
	cdef int feature_offset = 3 + 3

	cdef np.ndarray pair = np.zeros((pair_shape), dtype=DTYPE_DOUBLE)
	cdef np.ndarray new_Q, errors, p1, p2, p12, ranges
	cdef double error, i
	cdef int min_id

	pair[1] = v1
	pair[2] = v2

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
	pair[0] = errors[min_id]
	i = min_id * 0.1
	pair[target_offset: feature_offset] = (i * p1 + (1 - i) * p2)[:3]

	if features.shape[0] != 0:
		pair[feature_offset:] = i * features[v1] + (1 - i) * features[v2]

	return pair

cdef np.ndarray make_homogeneous(np.ndarray arr):
	"""appends an array with [1] to make it homogeneous.

	Args:
		arr (:class:`ndarray`): 1d vector

	:rtype: :class:`ndarray`"""
	return np.hstack([arr, 1])
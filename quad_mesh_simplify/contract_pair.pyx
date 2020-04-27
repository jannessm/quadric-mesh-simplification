import numpy as np
cimport numpy as np
from pair cimport Pair

cpdef (
	np.ndarray,
	np.ndarray,
	np.ndarray
) contract_first_pair(
	list pairs,
	np.ndarray positions,
	np.ndarray Q,
	np.ndarray features=None
):
	p = pairs[0]
	cdef long v1 = p.v1
	cdef long v2 = p.v2

	# update positions
	positions[v1] = np.copy(p.target)
	positions = np.delete(positions, v2, 0)

	# update Q
	Q[v1] = Q[v1] + Q[v2]
	Q = np.delete(Q, v2, 0)

	# update features
	if features is not None and p:
		features[v1] = p.new_features
		features = np.delete(features, v2, 0)

	return positions, Q, features

import numpy as np
cimport numpy as np

DTYPE_LONG = np.int64

ctypedef np.int64_t DTYPE_LONG_T

from utils cimport get_faces_for_node
from utils cimport get_rows

cpdef np.ndarray compute_valid_pairs(np.ndarray positions, np.ndarray face, double threshold):
	"""computes all valid pairs. A valid pair of nodes is either nodes that are connected by an edge, or their distance are below the given threshold.
	The returned array is of shape (num_pairs, 2) containing the ids of all nodes.

	Args:
		positions (:class:`ndarray`): positions of nodes with shape (num_nodes, 3)
		face (:class:`ndarray`): face of mesh containing node ids with shape (num_faces, 3)
		threshold (:class:`double`, optional): if provided, includes pairs that have a smaller distance than this threshold. (default: :obj:`0`)

	:rtype: :class:`ndarray`
	"""
	cdef np.ndarray valid_pairs = np.zeros((0,2), dtype=DTYPE_LONG)

	# option 1 to be valid
	cdef np.ndarray f, edges

	for f in face:
		edges = np.array([
			[f[0], f[1]],
			[f[1], f[2]],
			[f[2], f[0]]
		], dtype=DTYPE_LONG)

		valid_pairs = np.vstack([valid_pairs, edges])
		valid_pairs = remove_duplicates(valid_pairs)

	# option 2: distance below threshold
	cdef long i
	cdef np.ndarray distance, pairs, rows
	
	if threshold is not None:
		for i in range(positions.shape[0]):
			distance = np.linalg.norm(positions - positions[i], axis=1)
			
			pairs = get_rows(distance < threshold)
			# remove self-loops
			pairs = pairs[get_rows(pairs != i)]
			pairs = np.insert(pairs[:, None], 1, i, axis=1)

			valid_pairs = np.vstack([valid_pairs, pairs])
			valid_pairs = remove_duplicates(valid_pairs)

	return valid_pairs

cdef np.ndarray remove_duplicates(np.ndarray arr):
	"""removes duplicates in an array of edges.
	
	Args:
		arr (:class:`ndarray`): array

	:rtype: :class:`ndarray`"""
	return np.unique(np.sort(arr, axis=1), axis=0)
import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

from cython.parallel import prange

from utils import get_faces_for_node

cdef compute_Q(np.ndarray positions, np.ndarray face):
	r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used as an error value of this node.

	Args:
    positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
    face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

	:rtype: :class:`ndarray`"""

	cdef long num_nodes = positions.shape[0]
	cdef np.ndarray Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)

	cdef np.ndarray K, u, v, n, p

	
	cdef int i, j
	for i in range(positions.shape[0]):
		K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
		
		for face in get_faces_for_node(i, face):
			u = positions[face[0]]
			v = positions[face[1]]

			# calculate normal
			n = np.cross(u, v)
			n = n / np.norm(n)

			d = -(n * u).sum()

			p = np.hstack([n, d])

			K += p * p.T

		Q[j] = positions[i].T * K * positions[i]

	return Q
import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

from utils cimport get_faces_for_node

cpdef np.ndarray compute_Q(np.ndarray positions, np.ndarray face):
	r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used as an error value of this node.

	Args:
    positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
    face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

	:rtype: :class:`ndarray`"""

	cdef long num_nodes = positions.shape[0]
	cdef np.ndarray Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)

	cdef np.ndarray K, u, v, w, n, p, f
	cdef double d

	cdef long i
	for i in range(positions.shape[0]):
		K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
		
		for f in get_faces_for_node(i, face):
			u = positions[f[0]]
			v = positions[f[1]]
			w = positions[f[2]]

			# calculate normal
			n = np.cross(v - u, w - u)
			n /= np.linalg.norm(n)

			d = -(n * u).sum()

			p = np.hstack([n, d])

			K += p.dot(p.T)

		Q[i] = K

	return Q
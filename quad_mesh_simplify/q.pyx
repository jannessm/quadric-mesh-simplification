import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

from .utils cimport get_faces_for_node, face_normal

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] compute_Q(
	np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
	np.ndarray[DTYPE_LONG_T, ndim=2] face):
	r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used for calculating error values.

	The error is later calculated by (v.T Q v) and forms the quadric error metric.

	Args:
	    positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
	    face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

	:rtype: :class:`ndarray`"""

	cdef np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q
	cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] K, p
	cdef np.ndarray[DTYPE_LONG_T, ndim=1] f
	cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] u, v, w, n
	cdef double d
	cdef long num_nodes, i

	num_nodes = positions.shape[0]
	Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)

	for i in range(positions.shape[0]):
		K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
		
		for f in get_faces_for_node(i, face):
			u = positions[f[0]]

			n = face_normal(positions[f], True, -1)

			d = -(n * u).sum()

			p = np.hstack([n, d])[:, None]

			K += p.dot(p.T)

		Q[i] = K

	return Q
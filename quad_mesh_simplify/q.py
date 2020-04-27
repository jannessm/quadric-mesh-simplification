import numpy as np

DTYPE_DOUBLE = np.double

from . import utils

def compute_Q(positions, face):
	r"""computes for all nodes in :obj:`positions` a 4 x 4 matrix Q used as an error value of this node.

	Args:
    positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
    face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face

	:rtype: :class:`ndarray`"""

	num_nodes = positions.shape[0]
	Q = np.zeros((num_nodes, 4, 4), dtype=DTYPE_DOUBLE)

	for i in range(positions.shape[0]):
		K = np.zeros((4, 4), dtype=DTYPE_DOUBLE)
		
		for face in utils.get_faces_for_node(i, face):
			u = positions[face[0]]
			v = positions[face[1]]
			w = positions[face[2]]

			# calculate normal
			n = np.cross(v - u, w - u)
			n /= np.linalg.norm(n)

			d = -(n * u).sum()

			p = np.hstack([n, d])

			K += p * p.T

		Q[i] = K

	return Q
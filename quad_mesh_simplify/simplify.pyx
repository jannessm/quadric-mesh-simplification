import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array
from cpython.exc cimport PyErr_CheckSignals
import array

cdef extern from "c.simplify.h":
    cpdef simplify_mesh_c(np.ndarray positions, np.ndarray face, np.ndarray features, unsigned int num_nodes, double threshold)

def simplify_mesh(positions, face, num_nodes, features=None, threshold=0.):
    r"""simplify a mesh by contracting edges using the algortihm from `"Surface Simplification Using Quadric Error Metrics"
    <http://mgarland.org/files/papers/quadrics.pdf>`_.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        num_nodes (number): number of nodes that the final mesh will have
        threshold (number, optional): threshold of vertices distance to be a valid pair

    :rtype: (:class:`ndarray`, :class:`ndarray`)
    """

    if features == None:
        features = np.zeros((position.shape[0], 0))

    simplify_mesh_c(positions, face, features, num_nodes, threshold)

    return positions, face, features
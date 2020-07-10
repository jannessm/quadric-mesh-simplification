import numpy as np
import sys

cdef extern from "c/simplify.h":
    cdef tuple simplify_mesh_c(positions, face, features, unsigned int num_nodes, double threshold)

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

    # check types
    if not type(positions) == np.ndarray:
        raise Exception('positions has to be an ndarray.')
    if not positions.shape[1] == 3:
        raise Exception('positions has to be of shape N x 3.')
    if not positions.dtype == np.double:
        raise Exception('positions has to be of type double')

    if not type(face) == np.ndarray:
        raise Exception('face has to be an ndarray.')
    if not face.shape[1] == 3:
        raise Exception('face has to be of shape N x 3.')
    if not face.dtype == np.uint32:
        raise Exception('face has to be of type unsigned int (np.uint32)')

    if features is None:
        features = np.zeros((positions.shape[0], 0), np.double)
    if not type(features) == np.ndarray:
        raise Exception('features has to be an ndarray.')
    if not features.shape[0] == positions.shape[0]:
        raise Exception('first dimensions of features has to match first shape of positions.')
    if not features.dtype == np.double:
        raise Exception('features has to be of type double')

    if (positions.shape[0] ** 2 + positions.shape[0]) / 2 > sys.maxsize * 2:
        raise Exception('too many vertices. cannot build edge matrix.')

    new_pos = None
    new_face = None
    new_features = None
    if num_nodes < positions.shape[0] and features.shape[1] > 0:
        return simplify_mesh_c(positions, face, features, num_nodes, threshold)
    else:
        return simplify_mesh_c(positions, face, features, num_nodes, threshold)
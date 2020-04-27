import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

from pair import Pair
from q import compute_Q
from targets import compute_targets
from valid_pairs import compute_valid_pairs

cpdef simplify_mesh(positions, face, num_nodes, threshold=None):
    r"""simplify a mesh by contracting edges using the algortihm from `"Surface Simplification Using Quadric Error Metrics"
    <http://mgarland.org/files/papers/quadrics.pdf>`_.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        num_nodes (number): number of nodes that the final mesh will have
        threshold (number, optional): threshold of vertices distance to be a valid pair

    :rtype: (:class:`ndarray`, :class:`ndarray`)
    """

    # 1. compute Q for all vertices
    cdef np.ndarray Q = compute_Q(positions, face)

    # 2. Select valid pairs
    cdef np.ndarray valid_pairs = compute_valid_pairs(positions, face, threshold)

    # 3. compute optimal contration targets
    cdef np.ndarray errors
    pairs = []
    
    errors, pairs = compute_targets(positions, Q)

    # 4. create head sorted by costs
    errors = sort_errors(errors)

    # 5. contract vertices until num_nodes reached
    cdef double error
    while Q.shape[0] > num_nodes and errors.shape[0] != 0:
        p = pairs[errors[0]]

        # remove first row from 'heap'
        errors = np.delete(errors, (0), axis=0)

        # contract

        # update errors

        # sort errors
        errors = sort_errors(errors)

    return positions, face

cpdef sort_errors(np.ndarray arr):
    return np.sort(
            arr.view('float,float'),
            order=['f0'],
            axis=0
        ).view(DTYPE_DOUBLE)
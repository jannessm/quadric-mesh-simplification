import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

from pair cimport Pair
from q cimport compute_Q
from targets cimport compute_targets
from valid_pairs cimport compute_valid_pairs

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

    # 1. compute Q for all vertices
    cdef np.ndarray Q = compute_Q(positions, face)

    # 2. Select valid pairs
    cdef np.ndarray valid_pairs = compute_valid_pairs(positions, face, threshold)

    # 3. compute optimal contration targets
    cdef np.ndarray pairs
    
    pairs = compute_targets(positions, Q, valid_pairs, features)

    # 4. create head sorted by costs
    pairs = sort_pairs(pairs)

    # 5. contract vertices until num_nodes reached
    cdef double error
    cdef long v1
    cdef long v2
    cdef Pair p
    while positions.shape[0] > num_nodes and pairs.length > 0:
        p = pairs[0]
        v1 = p.v1
        v2 = p.v2

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

        # remove p
        pairs.remove(p)

        # update all other valid pairs
        for p in pairs:
            if p.v1 == v1:
                p.calculate_error(v1, p.v2, positions, Q, features)
            elif p.v1 == v2:
                p.calculate_error(v2, p.v2, positions, Q, features)
            elif p.v2 == v1:
                p.calculate_error(p.v1, v1, positions, Q, features)
            elif p.v2 == v2:
                p.calculate_error(p.v1, v2, positions, Q, features)

        # sort by errors
        pairs = sort_pairs(pairs)

    return positions, face

cdef list sort_pairs(list pairs):
    return pairs.sort(key=compare_by)

cdef double compare_by(p):
    return p.error
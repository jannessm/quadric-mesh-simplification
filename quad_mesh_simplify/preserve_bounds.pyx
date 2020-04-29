import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef void preserve_bounds(
    np.ndarray[DTYPE_DOUBLE_T, ndim=2] positions,
    np.ndarray[DTYPE_LONG_T, ndim=2] face,
    np.ndarray[DTYPE_DOUBLE_T, ndim=3] Q):
    """This method adds a large penality to the current Q matrix for each node of edge that is only part of one face and therefore forms a boundary.

    Note that it manipulates the matrix Q in place.

    Args:
        positions (:class:`ndarray`): array of shape num_nodes x 3 containing the x, y, z position for each node
        face (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
        Q (:class:`ndarray`): array of shape num_faces x 3 containing the indices for each triangular face
    """
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] edges, K, p
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] e, u, v, w, n
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] counts
    cdef long i, v1, v2
    cdef double d

    edges = np.zeros((0, 2 + 3), dtype=DTYPE_DOUBLE)

    # create edges
    for f in face:
        # calculate normal
        u = positions[f[0]]
        v = positions[f[1]]
        w = positions[f[2]]

        n = np.cross(v - u, w - u)
        n /= np.linalg.norm(n)
    
        edges = np.vstack([
            edges,
            np.array([
                [f[0], f[1], n[0], n[1], n[2]],
                [f[1], f[2], n[0], n[1], n[2]],
                [f[2], f[0], n[0], n[1], n[2]]
            ], dtype=DTYPE_DOUBLE)
        ])


    edges[:, :2] = np.sort(edges[:, :2], axis=1)
    edges.view('double, double, double, double, double').sort(order=['f0'], axis=0)

    i = 0
    while i < edges.shape[0]:
        e = edges[i]
        if i + 1 != edges.shape[0] and (e[:2] == edges[i+1, :2]).sum() == 2:
            # do not cover edge again
            i += 2
            continue
        else:
            v1 = <long>e[0]
            v2 = <long>e[1]

            # calculate penalties
            u = positions[v1]
            v = positions[v2]

            # calculate normal
            n = np.cross(u - e[2:], v - e[2:])
            n /= np.linalg.norm(n)
        
            d = -(n * u).sum()

            p = np.hstack([n, d])[:, None]

            K = p.dot(p.T) * 10e6
            Q[v1] += K
            Q[v2] += K
            i += 1

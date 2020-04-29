import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long
cimport cython

from .utils cimport get_rows

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
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=2] K, p
    cdef np.ndarray[DTYPE_DOUBLE_T, ndim=1] u, v, w, n1, n2
    cdef np.ndarray[DTYPE_LONG_T, ndim=1] counts, f, e
    cdef np.ndarray[DTYPE_LONG_T, ndim=2] edges
    cdef int i, j
    cdef double d

    # create edges
    edges = np.vstack([
        face[:, :2],
        face[:, 1:],
        face[:, [0,2]]
    ]).astype(DTYPE_LONG)

    print('created edges')
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    print('sorted edges')

    for i, e in enumerate(edges):
        if counts[i] > 1:
            continue
        
        # calculate face normal
        for f in face:
            if e[0] in f and e[1] in f:
                u, v, w = positions[f]
                n1 = np.cross(u - w, v - w)
                n1 /= np.linalg.norm(n1)
                break

        # calculate penalties
        u, v = positions[e]
        # calculate normal
        n2 = np.cross(u - n1, v - n1)
        n2 /= np.linalg.norm(n2)
        d = -n2.dot(u)

        p = np.hstack([n2, d])[:, None]

        K = p.dot(p.T) * 10e3
        Q[e] += K

import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double
DTYPE_LONG = np.long

ctypedef np.double_t DTYPE_DOUBLE_T
ctypedef np.long_t DTYPE_LONG_T

cpdef np.ndarray preserve_bounds(np.ndarray positions, np.ndarray face, np.ndarray Q):
	cdef np.ndarray edges = np.zeros((0,2 + 3), dtype=DTYPE_DOUBLE)
	cdef np.ndarray e, counts, K, u, v, w, n, p
	cdef long i, v1, v2
	cdef double d

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
			v1 = e[0]
			v2 = e[1]

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


	return Q
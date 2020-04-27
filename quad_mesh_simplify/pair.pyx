import numpy as np
cimport numpy as np

DTYPE_DOUBLE = np.double

ctypedef np.double_t DTYPE_DOUBLE_T

cdef class Pair:
	r"""managing a valid pair by containing the ids of the nodes and the new contracted node position"""

	cdef public double error
	cdef public long v1, v2

	cdef public np.ndarray target, new_features

	cpdef Pair calculate_error(self, v1, v2, positions, Q, features=None):
		cdef np.ndarray new_Q, errors, p1, p2, p12, ranges
		cdef double error, i
		cdef int min_id

		self.v1 = v1
		self.v2 = v2

		new_Q = Q[v1] + Q[v2]
		
		# do not use explicit solution because of feature trade-off
		errors = np.zeros((0), dtype=DTYPE_DOUBLE)
		p1 = self.make_homogeneous(positions[v1])
		p2 = self.make_homogeneous(positions[v2])

		# calculate errors for a 10 different targets on p1 -> p2
		r = np.arange(0, 1.1, 0.1)[:, None]
		p12 = r.dot(p1[:, None].T) + (1 - r).dot(p2[:, None].T)
		errors = p12.dot(new_Q).dot(p12.T) * np.eye(11)
		errors = np.max(errors, axis=1)

		# get minimal error
		min_id = np.argmin(errors)
		self.error = errors[min_id]
		i = min_id * 0.1
		self.target = (i * p1 + (1 - i) * p2)[:3]

		if features is not None:
			self.feature = i * features[v1] + (1 - i) * features[v2]

		return self

	def __repr__(self):
		return 'Pair {}, {}\n  error: {}\n  target: {}\n'.format(
			self.v1,
			self.v2,
			self.error,
			self.target
		)

	def __eq__(self, other):
		return (
			self.v1 == other.v1 and
			self.v2 == other.v2 and
			self.error == other.error
		)

	cdef np.ndarray make_homogeneous(self, np.ndarray arr):
		return np.hstack([arr, 1])
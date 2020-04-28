import numpy as np
cimport numpy as np

cpdef np.ndarray get_faces_for_node(long node_id, np.ndarray face):
	"""returns all faces that contain node_id

	Args:
		node_id (:class:`long`): node id
		face (:class:`ndarray`): array (num_faces, 3) containing all faces for a mesh

	:rtype: :class:`ndarray`
	"""
	cdef np.ndarray rows = get_rows(face == node_id)
	return face[rows]

cpdef np.ndarray get_rows(np.ndarray condition):
	"""returns all unique rows where a given condition is at least one time true.

	Args:
		condition (:class:`ndarray`): boolean array

	:rtype: :class:`ndarray`"""
	return np.unique(np.where(condition)[0])
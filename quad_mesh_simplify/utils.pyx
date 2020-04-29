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

cpdef np.ndarray face_normal(np.ndarray pos, int normalized, int reference_id):
	"""returns a face normal

	Args:
		pos (:class:`ndarray):` position of face nodes (shape (3, 3)). Each row is represents a node.

	:rtype: :class:`ndarray`"""

	cdef np.ndarray n

	if reference_id > -1:
		n = np.cross(
			pos[(1 + reference_id) % 3] - pos[reference_id],
			pos[(2 + reference_id) % 3] - pos[reference_id]
		)
	else:
		n = np.cross(pos[1] - pos[0], pos[2] - pos[0])
	
	if normalized and np.linalg.norm(n) != 0:
		n /= np.linalg.norm(n)
	
	return n
# cpdef void clean_mesh(
#     double[:, :] pos,
#     long [:, :] face,
#     unsigned char [:] deleted_pos
#     unsigned char [:] deleted_faces):

# # delete positions, faces and features
#     pos__ = pos_[[ not deleted for deleted in deleted_pos_.tolist()]]
#     face__ = face_[[not deleted for deleted in deleted_face_.tolist()]]
#     face_view = face__
#     sum_diminish_ = array.array('i', [])
#     sum_diminish_ = array.clone(sum_diminish_, deleted_pos.shape[0], True)
#     sum_diminish = sum_diminish_
#     diminish_by = 0
    
#     for i in range(deleted_pos.shape[0]):
#         sum_diminish[i] += diminish_by
        
#         # if node was deleted diminish next nodes by 1
#         if deleted_pos[i]:
#             diminish_by += 1

#     for i in range(face_view.shape[0]):
#         for j in range(face_view.shape[1]):
#             face_view[i, j] -= sum_diminish[face_view[i, j]]

cpdef void delete_orphans(
    long [:,:] face,
    char [:] deleted_positions,
    char [:] deleted_faces):

    cdef int i, j

    for i in range(deleted_positions.shape[0]):
        deleted_positions[i] = True

    for i in range(face.shape[0]):
        if deleted_faces[i]:
            continue

        for j in range(3):
            deleted_positions[face[i, j]] = False

cpdef int count(char [:] deleted):
    cdef int i, deleted
    deleted = 0
    for i in range(deleted.shape[0]):
        deleted += 1
    return deleted
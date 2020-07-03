from torch_geometric.io import read_obj, read_ply
from quad_mesh_simplify import simplify_mesh
from mayavi import mlab
from time import time

lion = read_ply('./test_data/Lion.ply')
print(lion)

pos = lion.pos
face = lion.face
mlab.triangular_mesh(
      pos[:,0], pos[:,1], pos[:,2],
      face.t()
    )
mlab.show()

start = time()
res_pos, res_face = simplify_mesh(pos.numpy().astype('double'), face.numpy().T.astype('uint32'), 100)
print('needed',time() - start, 'sec')

mlab.triangular_mesh(
      res_pos[:,0], res_pos[:,1], res_pos[:,2],
      res_face
    )
mlab.show()
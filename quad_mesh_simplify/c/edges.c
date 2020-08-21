#include "edges.h"

UpperTriangleMat* create_edges(Mesh* mesh) {
  unsigned int i, j, v1, v2;
  char val;
  UpperTriangleMat* edges = upper_zeros(mesh->n_vertices);

  // create edges
  for (i = 0; i < mesh->n_face; i++) {
    for (j = 0; j < 3; j++) {      
      v1 = mesh->face[i * 3 + j];
      v2 = mesh->face[i * 3 + ((j + 1) % 3)];
      val = upper_get(edges, v1, v2);
      upper_set(edges, v1, v2, val + 1);
    }
  }

  return edges;
}
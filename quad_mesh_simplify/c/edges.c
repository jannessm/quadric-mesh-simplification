#include "edges.h"

UpperTriangleMat* create_edges(Mesh* mesh) {
  unsigned int i, j, a, v1, v2;
  UpperTriangleMat* edges = upper_zeros(mesh->n_vertices, mesh->n_vertices);

  // create edges
  for (i = 0; i < mesh->n_face; i++) {
    for (j = 0; j < 3; j++) {
      a = (j + 1) % 3;
      v1 = mesh->face[i * 3 + j] < mesh->face[i * 3 + a] ? mesh->face[i * 3 + j] : mesh->face[i * 3 + a];
      v2 = mesh->face[i * 3 + j] == v1 ? mesh->face[i * 3 + a] : mesh->face[i * 3 + j];

      // edge was not seen before
      if (upper_get(edges, v1, v2) == 0) {
        upper_set(edges, v1, v2, 1);
      } else {
        upper_set(edges, v1, v2, upper_get(edges, v1, v2) + 1);
      }
    }
  }

  return edges;
}
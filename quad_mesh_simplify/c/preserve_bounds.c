#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "maths.h"
#include "preserve_bounds.h"


void preserve_bounds(Mesh* mesh, double* Q, UpperTriangleMat* edges) {
  unsigned int i, j, k, a, v1, v2, v3;

  double *pos1, *pos2, *pos3, *p, *K, *n;

  // add penalties
  for (i = 0; i < mesh->n_face; i++) {
    for (j = 0; j < 3; j++) {
      a = (j + 1) % 3;
      // order v1, v2 for edge check
      v1 = mesh->face[i * 3 + j] < mesh->face[i * 3 + a] ? mesh->face[i * 3 + j] : mesh->face[i * 3 + a];
      v2 = mesh->face[i * 3 + j] == v1 ? mesh->face[i * 3 + a] : mesh->face[i * 3 + j];

      // edge was seen once and therefore is a border of the mesh
      if (upper_get(edges, v1, v2) == 1) {
        // do not order v1, v2
        v1 = mesh->face[i * 3 + j % 3];
        v2 = mesh->face[i * 3 + (j + 1) % 3];
        v3 = mesh->face[i * 3 + (j + 2) % 3];
        
        pos1 = &(mesh->positions[v1 * 3]);
        pos2 = &(mesh->positions[v2 * 3]);
        pos3 = &(mesh->positions[v3 * 3]);

        n = normal(pos1, pos2, pos3);
        p = normal(pos1, pos2, n);
        K = calculate_K(p);

        for (k = 0; k < 16; k++) {
          K[k] *= 10e3;
        }
        
        add_K_to_Q(&(Q[v1 * 16]), K);
        add_K_to_Q(&(Q[v2 * 16]), K);

        free(n);
        free(p);
        free(K);
      }
    }
  }
}
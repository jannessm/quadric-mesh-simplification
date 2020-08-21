#include <stdlib.h>
#include <stdio.h>
#include "valid_pairs.h"
#include "maths.h"

Array2D_uint* compute_valid_pairs(Mesh* mesh, UpperTriangleMat* edges, double threshold) {
  unsigned int i, j, k;
  unsigned int pair[2];
  double distance[3];
  Array2D_uint* pairs = array_zeros(0, 2);

  for (i = 0; i < mesh->n_vertices; i++) {
    for (j = i + 1; j < mesh->n_vertices; j++) {
      if (upper_get(edges, i, j) > 0) {
        pair[0] = i;
        pair[1] = j;
        array_put_row(pairs, pair);
      } else if (threshold > 0) {
        for (k = 0; k < 3; k++) {
          distance[k] = mesh->positions[i * 3 + k] - mesh->positions[j * 3 + k];
        }

        if (norm(distance) < threshold) {
          pair[0] = i;
          pair[1] = j;
          array_put_row(pairs, pair);
        }
      }
    }
  }

  return pairs;
}
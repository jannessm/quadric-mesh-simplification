#include "mesh.h"
#include "maths.h"
#include "utils.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

double* compute_Q(Mesh* mesh) {
  double *K, *p, *Q;
  double *pos1, *pos2, *pos3;
  unsigned int i;

  Q = calloc(mesh->n_vertices * 16, sizeof(double));

  for (i = 0; i < mesh->n_face; i++) {
    pos1 = &(mesh->positions[mesh->face[i*3] * 3]);
    pos2 = &(mesh->positions[mesh->face[i*3 + 1] * 3]);
    pos3 = &(mesh->positions[mesh->face[i*3 + 2] * 3]);

    p = normal(pos1, pos2, pos3);
    K = calculate_K(p);

    //update Q
    add_K_to_Q(&(Q[mesh->face[i * 3] * 16]), K);
    add_K_to_Q(&(Q[mesh->face[i * 3 + 1] * 16]), K);
    add_K_to_Q(&(Q[mesh->face[i * 3 + 2] * 16]), K);
    
    free(K);
    free(p);
  }
  
  return Q;
}
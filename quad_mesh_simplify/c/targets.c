#include <float.h>
#include <stdlib.h>
#include "array.h"
#include "mesh.h"
#include "pair.h"
#include "maths.h"
#include "targets.h"

PairList* compute_targets(Mesh* mesh, double* Q, Array2D_uint* valid_pairs) {
  PairList* pairs = pairlist_init();
  unsigned int i;
  Pair* p;

  #pragma omp parallel for shared(mesh, Q, valid_pairs, pairs) private(i, p)
  for (i = 0; i < valid_pairs->rows; i++) {
    p = calculate_pair_attributes(
        mesh, Q,
        valid_pairs->data[i * valid_pairs->columns],
        valid_pairs->data[i * valid_pairs->columns + 1]);
    #pragma omp critical
    pairlist_append(pairs, p);
  }

  return pairs;
}

Pair* calculate_pair_attributes(Mesh* mesh, double* Q, unsigned int v1, unsigned int v2) {
  printf("calc attr for %d & %d\n", v1, v2);
  Pair* pair = pair_init(mesh->feature_length);

  double *p1, *p2, p12[3], p112[3], new_Q[16];
  unsigned int i, j, min_id;
  double min_error, err;
  
  min_error = DBL_MAX;

  pair->v1 = v1;
  pair->v2 = v2;

  p1 = &(mesh->positions[v1 * 3]);
  p2 = &(mesh->positions[v2 * 3]);

  for (i = 0; i < 3; i++) {
    p12[i] = (p2[i] - p1[i]) * 0.1;
  }

  for (i = 0; i < 16; i++) {
    new_Q[i] = Q[v1 * 16 + i] + Q[v2 * 16 + i];
  }

  min_id = 0;
  for (i = 0; i < 11; i++) {
    for (j = 0; j < 3; j++) {
      p112[j] = p1[j] + p12[j] * i;
    }

    err = error(p112, new_Q);

    if (err <= min_error) {
      min_error = err;
      min_id = i;
      for (j = 0; j < 3; j++) {
        pair->target[j] = p112[j];
      }
    }
  }

  pair->error = min_error;

  if (mesh->feature_length > 0) {
    for (i = 0; i < mesh->feature_length; i++) {
      pair->feature[i] = mesh->features[v1 * mesh->feature_length + i] * min_id * 0.1 +
                          mesh->features[v2 * mesh->feature_length + i] * (1 - min_id * 0.1);
    }
  }

  return pair;
}
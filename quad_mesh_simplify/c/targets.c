#include <float.h>
#include <stdlib.h>
#include "array.h"
#include "mesh.h"
#include "pair.h"
#include "maths.h"
#include "targets.h"
#include <stdio.h>

PairList* compute_targets(Mesh* mesh, double* Q, Array2D_uint* valid_pairs) {
  PairList* pairs = pairlist_init();
  unsigned int i;
  Pair* p;

  for (i = 0; i < valid_pairs->rows; i++) {
    p = calculate_pair_attributes(
        mesh, Q,
        valid_pairs->data[i * valid_pairs->columns],
        valid_pairs->data[i * valid_pairs->columns + 1]);
    pairlist_append(pairs, p);
  }

  return pairs;
}

Pair* calculate_pair_attributes(Mesh* mesh, double* Q, unsigned int v1, unsigned int v2) {
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
    err = calc_error(p112, new_Q);

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

    // DEBUG
    double n1 = norm(&mesh->features[v1 * mesh->feature_length]);
    double n2 = norm(&mesh->features[v2 * mesh->feature_length]);
    double max = n1 > n2 ? n1 : n2;
    double n = norm(pair->feature);
    if (n > max + 10e-7) {
      printf("new feat > 107:\n  min_id %d\n  %f %f %f: %f\n  v1 %d\n  %f %f %f\n  v2 %d\n  %f %f %f\n",
        min_id,
        pair->feature[0], pair->feature[1], pair->feature[2],
        n,
        v1,
        mesh->features[v1 * mesh->feature_length], mesh->features[v1 * mesh->feature_length + 1], mesh->features[v1 * mesh->feature_length + 2],
        v2,
        mesh->features[v2 * mesh->feature_length], mesh->features[v2 * mesh->feature_length + 1], mesh->features[v2 * mesh->feature_length + 2]
      );
      exit(-1);
    }
  }

  return pair;
}
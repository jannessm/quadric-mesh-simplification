#include <stdlib.h>
#include <stdbool.h>
#include "contract_pair.h"
#include "targets.h"

void update_pairs(
    PairHeap* heap,
    Mesh* mesh,
    double* Q,
    unsigned int v1,
    unsigned int v2
  ) {

  unsigned int i;
  Pair* pair;

  for (i = 1; i < heap->length; i++) {
    pair = heap_get_pair(heap, i);

    if ((pair->v1 == v1 || pair->v2 == v1) &&
        (pair->v1 == v2 || pair->v2 == v2)) {      
      pair->error = -10e6;
    
    } else if (pair->v1 == v1 || pair->v1 == v2) {      
      heap->nodes[i] = calculate_pair_attributes(mesh, Q, v1, pair->v2);
      pair_free(pair);
    
    } else if (pair->v2 == v1 || pair->v2 == v2) {
      heap->nodes[i] = calculate_pair_attributes(mesh, Q, pair->v1, v1);
      pair_free(pair);
    }
  }

  heap_build(heap);
}

void update_face(
    Mesh* mesh,
    bool* deleted_faces,
    unsigned int v1,
    unsigned int v2) {
  unsigned int i, j;
  bool v1_in_face, v2_in_face;

  for (i = 0; i < mesh->n_face; i++) {
    if (deleted_faces[i]) {
      continue;
    }

    v1_in_face = false;
    v2_in_face = false;

    for (j = 0; j < 3; j++) {
      if (mesh->face[i * 3 + j] == v1) {
        v1_in_face = true;
      }

      if (mesh->face[i * 3 + j] == v2) {
        v2_in_face = true;
        mesh->face[i * 3 + j] = v1;
      }
    }

    if (v1_in_face && v2_in_face) {
      deleted_faces[i] = true;
    }
  }
}


void update_features(
    Mesh* mesh,
    Pair* pair) {
  
  unsigned int i;

  for (i = 0; i < mesh->feature_length; i++) {
    mesh->features[pair->v1 * mesh->feature_length + i] = pair->feature[i];
  }
}
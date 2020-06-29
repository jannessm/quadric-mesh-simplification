#include <stdlib.h>
#include <stdbool.h>
#include "mesh.h"

void clean_positions_and_features(Mesh* mesh, bool* deleted_pos) {
  unsigned int new_size, i, j, new_i;
  
  new_size = 0;

  for (i = 0; i < mesh->n_vertices; i++) {
    if (!deleted_pos[i]) {
      new_size++;
    }
  }

  double* new_positions = malloc(new_size * 3 * sizeof(double));
  double* new_features = malloc(new_size * mesh->feature_length * sizeof(double));

  new_i = 0;

  for (i = 0; i < mesh->n_vertices; i++) {
    if (!deleted_pos[i]) {
      new_positions[new_i * 3] = mesh->positions[i * 3];
      new_positions[new_i * 3 + 1] = mesh->positions[i * 3 + 1];
      new_positions[new_i * 3 + 2] = mesh->positions[i * 3 + 2];

      for (j = 0; j < mesh->feature_length; j++) {
        new_features[new_i * mesh->feature_length + j] = mesh->features[i * mesh->feature_length + j];
      }

      new_i++;
    }
  }

  free(mesh->positions);
  free(mesh->features);
  mesh->positions = new_positions;
  mesh->features = new_features;
  mesh->n_vertices = new_size;
}

void clean_face(Mesh* mesh, bool* deleted_faces, bool* deleted_positions) {
  unsigned int i, j, new_i, diminish_by, new_size;
  unsigned int *sum_diminish, *new_face;
  
  diminish_by = 0;
  
  sum_diminish = calloc(mesh->n_vertices, sizeof(unsigned int));

  for (i = 0; i < mesh->n_vertices; i++) {
    sum_diminish[i] = diminish_by;
    if (deleted_positions[i]) {
      diminish_by++;
    }
  }

  new_size = 0;
  for (i = 0; i < mesh->n_face; i++) {
    if (!deleted_faces[i]) {
      new_size++;
    }
  }

  new_face = malloc(new_size * 3 * sizeof(unsigned int));

  new_i = 0;
  for (i = 0; i < mesh->n_face; i++) {
    if (!deleted_faces[i]) {
      for (j = 0; j < 3; j++) {
        new_face[new_i * 3 + j] = mesh->face[i * 3 + j] - sum_diminish[mesh->face[i * 3 + j]];
      }
      new_i++;
    }
  }

  free(mesh->face);
  free(sum_diminish);

  mesh->face = new_face;
  mesh->n_face = new_size;
}
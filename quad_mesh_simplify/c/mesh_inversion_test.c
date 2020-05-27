#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include "mesh_inversion.h"
#include "mesh.h"

void test_non_inverted_mesh() {
  const char* test_case = "non inverted mesh";
  unsigned int i;

  double positions[] = {
    0.25, 0, 0.,
    -.25, 0, 0.,
    0.5, .5, 1.,
    -.5, 0.5, 1.,
    0.75, 0., 0.
  };

  double new_pos[] = {
    .5, .5, 1.
  };

  unsigned int face[] = {
    0,2,3,
    0,1,3,
    0,4,2
  };

  unsigned int new_face[] = {
    0,2,3,
    0,0,3,
    0,4,2
  };

  bool *deleted_faces = calloc(3, sizeof(bool));

  Mesh m = {positions, NULL, face, 5, 3, 0};

  if (has_mesh_inversion(0, 1, &m, new_pos, deleted_faces)) {
    printf("✗ %s: should not have a mesh inversion\n", test_case);
    exit(-2);
  }

  free(deleted_faces);

  printf("✓ %s\n", test_case);
}

void test_inverted_mesh() {
  const char* test_case = "inverted mesh";
  unsigned int i;

  double positions[] = {
    0.25, 0, 0.,
    -.25, 0, 0.,
    0.5, .5, 1.,
    -.5, 0.5, 1.,
    0.75, 0., 0.
  };

  double new_pos[] = {
    -.25, 0, 0.
  };

  unsigned int face[] = {
    0,2,3,
    0,1,3,
    0,4,2
  };

  unsigned int new_face[] = {
    0,2,3,
    0,1,3,
    0,1,2
  };

  bool *deleted_faces = calloc(3, sizeof(bool));

  Mesh m = {positions, NULL, face, 5, 3, 0};

  if (!has_mesh_inversion(1, 4, &m, new_pos, deleted_faces)) {
    printf("✗ %s: should have a mesh inversion\n", test_case);
    exit(-2);
  }

  free(deleted_faces);

  printf("✓ %s\n", test_case);
}

int main(void) {
  test_non_inverted_mesh();
  test_inverted_mesh();
}
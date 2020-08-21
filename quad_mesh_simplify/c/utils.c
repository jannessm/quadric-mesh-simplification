#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mesh.h"

Mesh read_ply(char* filename) {
  FILE * fp;
  size_t len = 0;
  ssize_t read;
  char * line = NULL;
  
  unsigned int n_vertices = 0;
  unsigned int n_face = 0;
  unsigned int i;
  unsigned int* face;
  int face_nodes;
  double* positions;

  fp = fopen(filename, "r");
  if (fp == NULL)
    exit(EXIT_FAILURE);

  // process header
  while ((read = getline(&line, &len, fp)) != -1) {
    if (strstr(line, "element vertex") != NULL) {
      sscanf(line, "element vertex %u", &n_vertices);
    }

    if (strstr(line, "element face") != NULL) {
      sscanf(line, "element face %u", &n_face);
    }

    if (strstr(line, "end_header") != NULL) {
      break;
    }
  }

  if (n_vertices == 0 && n_face == 0) {
    fprintf(stderr, "Couldn't parse ply file");
    fclose(fp);
  }

  // alloc positions and face
  positions = malloc(sizeof(double) * n_vertices * 3);
  face = malloc(sizeof(unsigned int) * n_face * 3);

  // process vertices
  for (i = 0; i < n_vertices; i++) {
    getline(&line, &len, fp);
    sscanf(line, "%lf %lf %lf", &positions[i * 3], &positions[i * 3 + 1], &positions[i * 3 + 2]);
  }
  
  // process face
  for (i = 0; i < n_face; i++) {
    getline(&line, &len, fp);
    sscanf(line, "%d ", &face_nodes);
    if (face_nodes != 3) {
      fprintf(stderr, "ply file contains faces with more than 3 nodes\n");
      exit(-2);
    }

    sscanf(line, "3 %u %u %u", &face[i * 3], &face[i * 3 + 1], &face[i * 3 + 2]);
  }

  fclose(fp);
  if (line)
    free(line);

  Mesh m = {positions, NULL, face, n_vertices, n_face};

  return m;
}
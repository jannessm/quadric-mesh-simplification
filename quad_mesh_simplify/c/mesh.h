#ifndef INCLUDE_MESH
#define INCLUDE_MESH
  typedef struct Mesh {
    double* positions;
    double* features;
    unsigned int* face;
    unsigned int n_vertices;
    unsigned int n_face;
  } Mesh;
#endif
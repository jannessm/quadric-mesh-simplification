#ifndef INCLUDE_MESH
#define INCLUDE_MESH
  typedef struct Mesh {
    double* positions;
    double* features;
    unsigned int* face;
    unsigned long n_vertices;
    unsigned long n_face;
    unsigned long feature_length;
  } Mesh;
#endif
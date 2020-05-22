#include "utils.h"

int main(int argc, char *argv[]) {  
  if (argc < 2) {
    fprintf(stderr, "please provide a ply file\n");
    exit(EXIT_FAILURE);
  }

  double* positions;
  unsigned int* face;
  
  struct Mesh mesh = read_ply(argv[1]);

  printf("read %u verts and %u faces\n", mesh.n_vertices, mesh.n_face);
}
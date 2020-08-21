#include <stdbool.h>
#include <stdlib.h>
#include "preserve_bounds.h"
#include "utils.h"
#include "test_utils.h"
#include "mesh.h"
#include "array.h"
#include "edges.h"
#include "sparse_mat.h"
#include "valid_pairs.h"

void test_valid_edges() {
  const char* test_case = "valid pairs without threshold";

  double positions[] = {
    0., 0., 0.,
    1., 0., 0.,
    1., 1., 0.,
    1., 1., 1.,
    1., 0., 1.,
  };

  unsigned int face[] = {
    0, 1, 2,
    1, 2, 3
  };

  unsigned int pairs[] = {
    0, 1,
    0, 2,
    1, 2,
    1, 3,
    2, 3
  };

  Array2D_uint expected = {5, 2, pairs};

  Mesh m = {
    positions, NULL, face,
    sizeof(positions) / sizeof(double) / 3,
    sizeof(face)/sizeof(unsigned int) / 3
  };

  SparseMat* edges = create_edges(&m);

  Array2D_uint* result = compute_valid_pairs(&m, edges, 0);

  array_equal(test_case, &expected, result, 0, expected.rows * expected.columns);

  sparse_free(edges);
  array_free(result);

  printf("✓ %s\n", test_case);
}

void test_valid_pairs() {
  const char* test_case = "valid pairs with threshold";

  double positions[] = {
    0., 0., 0.,
    1., 0., 0.,
    1., 1., 0.,
    1., 1., 1.,
    1., 0., 1.,
  };

  unsigned int face[] = {
    0, 1, 2,
    1, 2, 3
  };

  unsigned int pairs[] = {
    0, 1,
    0, 2,
    0, 3,
    0, 4,
    1, 2,
    1, 3,
    1, 4,
    2, 3,
    2, 4,
    3, 4
  };

  Array2D_uint expected = {10, 2, pairs};

  Mesh m = {
    positions, NULL, face,
    sizeof(positions) / sizeof(double) / 3,
    sizeof(face)/sizeof(unsigned int) / 3
  };

  SparseMat* edges = create_edges(&m);

  Array2D_uint* result = compute_valid_pairs(&m, edges, 2);

  array_equal(test_case, &expected, result, 0, expected.rows * expected.columns);

  sparse_free(edges);
  array_free(result);

  printf("✓ %s\n", test_case);
}

int main(void) {
  test_valid_edges();
  test_valid_pairs();
}
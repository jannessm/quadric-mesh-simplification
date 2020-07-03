#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "upper_tri.h"

char upper_get(UpperTriangleMat* mat, unsigned int row, unsigned int column) {
  if (row > column) {
    return mat->values[(row * row + row) / 2 + column];
  } else {
    return mat->values[(column * column + column) / 2 + row];
  }
}

void upper_set(UpperTriangleMat* mat, unsigned int row, unsigned int column, char value) {
  if (row > column) {
    mat->values[(row * row + row) / 2 + column] = value;
  } else {
    mat->values[(column * column + column) / 2 + row] = value;
  }
}

UpperTriangleMat* upper_zeros(unsigned int size) {
  UpperTriangleMat* mat = malloc(sizeof(UpperTriangleMat));
  mat->values = calloc((size * size + size) / 2, sizeof(char));
  mat->columns = size;
  return mat;
}

void upper_free(UpperTriangleMat* mat) {
  free(mat->values);
  free(mat);
}

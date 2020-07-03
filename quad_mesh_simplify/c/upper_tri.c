#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "upper_tri.h"

char upper_get(UpperTriangleMat* mat, unsigned int row, unsigned int column) {
  if (row > column) {
    return mat->values[column * mat->columns - ((column * (column + 1)) / 2) + row];
  } else {
    return mat->values[row * mat->columns - ((row * (row + 1)) / 2) + column];
  }
}

void upper_set(UpperTriangleMat* mat, unsigned int row, unsigned int column, char value) {
  if (row > column) {
    mat->values[column * mat->columns - ((column * (column + 1)) / 2) + row] = value;
  } else {
    mat->values[row * mat->columns - ((row * (row + 1)) / 2) + column] = value;
  }
}

UpperTriangleMat* upper_zeros(unsigned int rows, unsigned int columns) {
  UpperTriangleMat* mat = malloc(sizeof(UpperTriangleMat));
  mat->values = calloc(rows * columns - ((rows * (rows + 1)) / 2) + columns, sizeof(char));
  mat->columns = columns;
  return mat;
}

void upper_free(UpperTriangleMat* mat) {
  free(mat->values);
  free(mat);
}

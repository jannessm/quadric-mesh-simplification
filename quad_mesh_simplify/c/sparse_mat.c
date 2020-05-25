#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "sparse_mat.h"

double sparse_get(SparseMat* mat, unsigned int row, unsigned int column) {
  unsigned int i, j;
  for (i = 0; i < mat->length; i++) {
    if (mat->rows[i] == row && mat->columns[i] == column) {
      return mat->values[i];
    }
  }
  return 0;
}

bool sparse_has_entry(SparseMat* mat, unsigned int row, unsigned int column) {
  unsigned int i, j;
  for (i = 0; i < mat->length; i++) {
    if (mat->rows[i] == row && mat->columns[i] == column) {
      return true;
    }
  }

  return false;
}

void sparse_set(SparseMat* mat, unsigned int row, unsigned int column, double value) {
  unsigned int i, j;

  // if exists overwrite value
  if (sparse_has_entry(mat, row, column)) {
    for (i = 0; i < mat->length; i++) {
      if (mat->rows[i] == row && mat->columns[i] == column) {
        mat->values[i] = value;
      }
    }

  // if not create a new entry at the end
  } else {
    mat->length++;
    mat->values = realloc(mat->values, mat->length * sizeof(double));
    mat->rows = realloc(mat->rows, mat->length * sizeof(unsigned int));
    mat->columns = realloc(mat->columns, mat->length * sizeof(unsigned int));
    
    mat->values[mat->length - 1] = value;
    mat->rows[mat->length - 1] = row;
    mat->columns[mat->length - 1] = column;
  }
}

SparseMat sparse_empty() {
  SparseMat mat = {NULL, NULL, NULL, 0};
  return mat;
}

void sparse_free(SparseMat* mat) {
  free(mat->values);
  free(mat->rows);
  free(mat->columns);
}

void print_sparse(SparseMat* mat) {
  unsigned int i;
  for (i = 0; i < mat->length; i++) {
    printf("%d, %d: %.4f\n", mat->rows[i], mat->columns[i], mat->values[i]);
  }
}
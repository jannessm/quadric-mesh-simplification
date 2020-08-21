#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "upper_tri.h"

unsigned long sum_formular(unsigned int i) {
  unsigned long i_ = (unsigned long) i;
  return (unsigned long) (i_ * i_ + i_) / 2;
}

unsigned long get_index(unsigned int row, unsigned int column) {
  if (row > column) {
    return sum_formular(row) + column;
  } else {
    return sum_formular(column) + row;
  } 
}

char upper_get(UpperTriangleMat* mat, unsigned int row, unsigned int column) {
  return mat->values[get_index(row, column)];
}

void upper_set(UpperTriangleMat* mat, unsigned int row, unsigned int column, char value) {
  mat->values[get_index(row, column)] = value;
}

UpperTriangleMat* upper_zeros(unsigned int size) {
  UpperTriangleMat* mat = malloc(sizeof(UpperTriangleMat));
  mat->values = calloc(sum_formular(size), sizeof(char));
  mat->columns = size;
  return mat;
}

void upper_free(UpperTriangleMat* mat) {
  free(mat->values);
  free(mat);
}

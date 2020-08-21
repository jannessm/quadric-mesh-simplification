#include <stdlib.h>
#include <stdio.h>
#include "array.h"

Array2D_uint* array_zeros(unsigned int rows, unsigned int columns) {
  Array2D_uint* array = malloc(sizeof(Array2D_uint));
  array->rows = rows;
  array->columns = columns;
  array->data = calloc(rows * columns, sizeof(unsigned int));
  return array;
}

void array_put_row(Array2D_uint* array, unsigned int* values) {
  unsigned int i;
  array->rows++;
  array->data = realloc(array->data, array->rows * array->columns * sizeof(unsigned int));
  for (i = 0; i < array->columns; i++) {
    array->data[(array->rows - 1) * array->columns + i] = values[i];
  }
}

void array_free(Array2D_uint* array) {
  free(array->data);
  free(array);
}

void print_array(Array2D_uint* array) {
  unsigned int i, j;
  for (i = 0; i < array->rows; i++) {
    for (j = 0; j < array->columns; j++) {
      printf("%u  ", array->data[i * array->columns + j]);
    }
    printf("\n");
  }
}
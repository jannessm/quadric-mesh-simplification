#ifndef INCLUDE_ARRAY
  #define INCLUDE_ARRAY
  typedef struct Array2D_uint {
    unsigned int rows;
    unsigned int columns;
    unsigned int* data;
  } Array2D_uint;
#endif

Array2D_uint* array_zeros(unsigned int rows, unsigned int columns);

void array_free(Array2D_uint* array);

void array_put_row(Array2D_uint* array, unsigned int* values);
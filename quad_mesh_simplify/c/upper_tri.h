#ifndef INCLUDE_SPARSE_MAT
#define INCLUDE_SPARSE_MAT
  typedef struct UpperTriangleMat {
    char* values;
    unsigned int columns;
  } UpperTriangleMat;
#endif

char upper_get(UpperTriangleMat* mat, unsigned int row, unsigned int column);

void upper_set(UpperTriangleMat* mat, unsigned int row, unsigned int column, char value);

UpperTriangleMat* upper_zeros(unsigned int rows, unsigned int columns);

void upper_free(UpperTriangleMat* mat);
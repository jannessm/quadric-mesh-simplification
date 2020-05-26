#ifndef INCLUDE_SPARSE_MAT
#define INCLUDE_SPARSE_MAT
  typedef struct SparseMat {
    double* values;
    unsigned int* rows;
    unsigned int* columns;
    unsigned int length;
  } SparseMat;
#endif

double sparse_get(SparseMat* mat, unsigned int row, unsigned int column);

char sparse_has_entry(SparseMat* mat, unsigned int row, unsigned int column);

void sparse_set(SparseMat* mat, unsigned int row, unsigned int column, double value);

SparseMat* sparse_empty();

void sparse_free(SparseMat* mat);

void print_sparse(SparseMat* mat);
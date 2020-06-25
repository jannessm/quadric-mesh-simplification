#include "array.h"

void print_array_double(double * arr, int rows, int cols);

void print_array_uint(unsigned int* arr, int rows, int cols);

void print_Q(double* q, unsigned int n_vertices, char to_stderr);

void q_equal(const char* test_case, double* expected, double* result, unsigned int from, unsigned int to);

void q_not_equal(const char* test_case, double* expected, double* result, unsigned int from, unsigned int to);

void array_equal(
  const char* test_case,
  Array2D_uint* expected, Array2D_uint* result,
  unsigned int from, unsigned to);
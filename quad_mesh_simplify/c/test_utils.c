#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "array.h"

void print_array_double(double * arr, int rows, int cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%f ", arr[i*cols+j]);
    }
    printf("\n");
  }
}

void print_array_uint(unsigned int* arr, int rows, int cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%u ", arr[i*cols+j]);
    }
    printf("\n");
  }
}

void print_Q(double* q, unsigned int from_vertex, unsigned int to_vertex, char to_stderr) {
  unsigned int i, j;
  for (i = from_vertex; i < to_vertex; i++) {
    for (j = 0; j < 4; j++) {
      if (to_stderr) {
        fprintf(stderr, "%.4lf  %.4lf  %.4lf  %.4lf\n", q[i * 16 + j*4], q[i * 16 + j*4 + 1], q[i * 16 + j*4 + 2], q[i * 16 + j*4 + 3]);
      } else {
        printf("%.4lf  %.4lf  %.4lf  %.4lf\n", q[i * 16 + j*4], q[i * 16 + j*4 + 1], q[i * 16 + j*4 + 2], q[i * 16 + j*4 + 3]);
      }
    }
    printf("\n");
  }
}

void print_Q_comparison(double* q1, double* q2, unsigned int from_vertex, unsigned int to_vertex, char to_stderr) {
  unsigned int i, j;
  for (i = from_vertex; i < to_vertex; i++) {
    for (j = i; j < i + 4; j++) {
      if (to_stderr) {
        fprintf(stderr, "%.4lf  %.4lf  %.4lf  %.4lf        ", q1[j*4], q1[j*4 + 1], q1[j*4 + 2], q1[j*4 + 3]);

        fprintf(stderr, "%.4lf  %.4lf  %.4lf  %.4lf\n", q2[j*4], q2[j*4 + 1], q2[j*4 + 2], q2[j*4 + 3]);
      } else {
        printf("%.4lf  %.4lf  %.4lf  %.4lf        ", q1[j*4], q1[j*4 + 1], q1[j*4 + 2], q1[j*4 + 3]);
        printf("%.4lf  %.4lf  %.4lf  %.4lf\n", q2[j*4], q2[j*4 + 1], q2[j*4 + 2], q2[j*4 + 3]);
      }
    }
    printf("\n");
  }
}

void q_equal(const char* test_case, double* expected, double* result, unsigned int from, unsigned int to) {
  unsigned int i;
  for (i = from; i < to; i++) {
    if (expected[i] - result[i] > 10e-6) {
      fprintf(stderr, "✗ %s:\nerror at value %d\nexpected:                             got:\n", test_case, i);
      print_Q_comparison(expected, result, from / 16, to / 16, true);
      exit(-2);
    }
  }
}

void q_not_equal(const char* test_case, double* expected, double* result, unsigned int from, unsigned int to) {
  unsigned int i;
  for (i = from; i < to; i++) {
    if (expected[i] - result[i] > 10e-6) {
      return;
    }
  }

  fprintf(stderr, "✗ %s:\nvalues from %d to %d are equal!\n", test_case, from, to);
  exit(-2);
}

void print_array_comparison(
  Array2D_uint* arr1, Array2D_uint* arr2,
  bool to_stderr) {

  unsigned int i, j, max_rows;
  max_rows = arr1->rows > arr2->rows ? arr1->rows : arr2->rows;

  for (i = 0; i < max_rows; i++) {

    for (j = 0; j < arr1->columns; j++) {
      if (to_stderr && i < arr1->rows) {
        fprintf(stderr, "%u  ", arr1->data[i * arr1->columns + j]);
      } else if (arr1->rows) {
        printf("%u  ", arr1->data[i * arr1->columns + j]);
      } else if (to_stderr) {
        fprintf(stderr, "   ");
      } else {
        printf("   ");
      }
    }

    if (to_stderr) {
      fprintf(stderr, "        ");
    } else {
      printf("        ");
    }
    
    for (j = 0; j < arr2->columns; j++) {
      if (to_stderr && i < arr2->rows) {
        fprintf(stderr, "%u  ", arr2->data[i * arr2->columns + j]);
      } else if (arr2->rows) {
        printf("%u  ", arr2->data[i * arr2->columns + j]);
      } else if (to_stderr) {
        fprintf(stderr, "   ");
      } else {
        printf("   ");
      }
    }
    printf("\n");
  }
}

void array_equal(
  const char* test_case,
  Array2D_uint* expected, Array2D_uint* result,
  unsigned int from, unsigned to) {

  if (expected->rows != result->rows || expected->columns != expected->columns) {
    fprintf(stderr, "✗ %s:\narrays doesnt have the same size\n", test_case);
    fprintf(stderr, "  expected (%d x %d)  got      (%d x %d)\n", expected->rows, expected->columns, result->rows, result->columns);
    print_array_comparison(expected, result, true);
    exit(0);
  }

  unsigned int i;
  for (i = from; i < to; i++) {
    if (expected->data[i] - result->data[i] > 10e-6) {
      fprintf(stderr, "✗ %s:\nerror at value %d\nexpected:     got:\n", test_case, i);
      print_array_comparison(expected, result, true);
      exit(0);
    }
  }
}

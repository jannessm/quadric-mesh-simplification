#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void print_Q(double* q, unsigned int from_vertex, unsigned int to_vertex, char to_stderr) {
  unsigned int i, j;
  for (i = from_vertex; i < to_vertex; i++) {
    for (j = i; j < i + 4; j++) {
      if (to_stderr) {
        fprintf(stderr, "%.4lf  %.4lf  %.4lf  %.4lf\n", q[j*4], q[j*4 + 1], q[j*4 + 2], q[j*4 + 3]);
      } else {
        printf("%.4lf  %.4lf  %.4lf  %.4lf\n", q[j*4], q[j*4 + 1], q[j*4 + 2], q[j*4 + 3]);
      }
    }
    printf("\n");
  }
}

void print_Q_comparision(double* q1, double* q2, unsigned int from_vertex, unsigned int to_vertex, char to_stderr) {
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
  int i;
  for (i = from; i < to; i++) {
    if (expected[i] - result[i] > 10e-6) {
      fprintf(stderr, "✗ %s:\nerror at value %d\nexpected:                             got:\n", test_case, i);
      print_Q_comparision(expected, result, from / 16, to / 16, true);
      exit(-2);
    }
  }
}

void q_not_equal(const char* test_case, double* expected, double* result, unsigned int from, unsigned int to) {
  int i;
  bool not_equal = true;
  for (i = from; i < to; i++) {
    if (expected[i] - result[i] > 10e-6) {
      return;
    }
  }

  fprintf(stderr, "✗ %s:\nvalues from %d to %d are equal!\n", test_case, from, to);
  exit(-2);
}
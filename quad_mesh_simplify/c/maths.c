#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "maths.h"

double* normal(double* v1, double* v2, double* v3) {
  int i, a, b;
  double * n, len;
  n = malloc(sizeof(double) * 4);
  
  for (i = 0; i < 3; i++) {
    a = (i + 1) % 3;
    b = (i + 2) % 3;
    n[i] = ((v1[a] - v2[a]) * (v3[b] - v2[b])) - 
           ((v1[b] - v2[b]) * (v3[a] - v2[a]));
  }

  len = norm(n);
  if (len > 0) {
    for (i = 0; i < 3; i++) {
      n[i] /= len;
    }
  }

  n[3] = - dot1d(n, v1);

  return n;
}

double norm(double* n) {
  double s = 0;
  
  int i;
  
  for (i = 0; i < 3; i++) {
    s += n[i] * n[i];
  }

  return sqrt(s);
}

double dot1d(double* v1, double* v2) {
  double s = 0;
  
  int i;
  
  for (i = 0; i < 3; i++) {
    s += v1[i] * v2[i];
  }

  return s;
}

void add_K_to_Q(double* A, double* B) {
  int i;
  for (i = 0; i < 16; i++) {
      A[i] += B[i];
  } 
}

double* calculate_K(double* p) {
  double* K = malloc(sizeof(double) * 16);
  int i, j;
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      K[i * 4 + j] = p[i] * p[j];
    }
  }
  return K;
}

double calc_error(double* p, double* Q) {
  int i, j;
  double err = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      err += p[j] * Q[j * 4 + i] * p[i];
    }
    err += Q[3 * 4 + i] * p[i];
  }

  for (i = 0; i < 3; i++) {
    err += p[i] * Q[i * 4 + 3];
  }
  err += Q[15];

  return err;
}
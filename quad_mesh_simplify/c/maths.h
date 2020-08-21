#ifndef INCLUDE_MATHS
#define INCLUDE_MATHS

double* normal(double* v1, double* v2, double* v3);

double norm(double* n);

double dot1d(double* v1, double* v2);

void add_K_to_Q(double* A, double* B);

double* calculate_K(double* p);

double calc_error(double* p, double* Q);
#endif
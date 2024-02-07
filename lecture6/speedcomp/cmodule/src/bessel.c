#include <bessel.h>

#define EPSILON 1e-8

void besselup_2d(double *z, int const N, double const ulim, int const l){
  for (int i=0; i<N; ++i){
    double x = (double) i * (ulim / N);
    if (fabs(x)<EPSILON){
      z[i*(l+1) + 0] = 1.;
      continue;
    }
    z[i*(l+1) + 0] = sin(x)/x;
    if (l == 0){
      continue;
    }
    z[i*(l+1) + 1] = sin(x)/(x*x) - cos(x)/x;
    if (l == 1){
      continue;
    }

    for (size_t j = 2; j<l+1; ++j){
      z[i*(l+1) + j] = (2*j - 1)/x * z[i*(l+1) + (j-1)] - z[i*(l+1) + (j-2)];
    }
  } 
}



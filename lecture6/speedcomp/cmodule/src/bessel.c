#include <stdio.h>
#include <bessel.h>

#define EPSILON 1e-8

void besselup_2d(double *z, int const N, double const ulim, int const l){
  int N_threads = atoi(getenv("OMP_NUM_THREADS"));
  int bms = N/N_threads; // bucket_min_size
  int overflow = N % N_threads;
  #pragma omp parallel for
  for (int p=0; p<N_threads; ++p){

    for (int i=p*bms; i<(p+1)*bms +(overflow * (p+1 == N_threads)); ++i){
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
}



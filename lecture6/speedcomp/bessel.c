#include <limits.h>
#include <time.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 1e-8

void besselup(double *restrict out, double const x, int const l){
/*
 * Make out[0]..out[l] equal to the bessel function of the 
 * 0..l'th kind on the value of x
 */

  out[0] = sin(x)/x;
  if (l == 0){
    return;
  }
  out[1] = sin(x)/(x*x) - cos(x)/x;
  if (l == 1){
    return;
  }
  for (size_t i = 2; i<l+1; ++i){
    out[i] = (2*i - 1)/x * out[i-1] - out[i-2];
  }
}

void besselup_2d(double *z, int const N, double const ulim, int const l){
/* #pragma omp parallel for */
  for (int i=0; i<N; ++i){
    double x = (double) i * (ulim / N);
    if (fabs(x)<EPSILON){
      z[i*(l+1) + 0] = 1;
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

int main(){
  int N = 1e6;

  int l = 50;
  double ulim = 50.;
  clock_t begin = clock();
  double *z = malloc(sizeof(double) * (l+1) * N);
  if (!z){
    printf("You cocked it\n");
    exit(1);
  }
  memset(z, 0, sizeof(double) * (l+1) * N);

  besselup_2d(z, N, ulim, l);
  /* for (int i=0; i<N; ++i){ */
  /*   double x = (double) i * (ulim / N); */
  /*   besselup(&z[i*(l+1)], x, l); */
  /* } */ 

  clock_t end = clock();
  double time =(double) (end - begin)/CLOCKS_PER_SEC;


#if defined(output)
  (void)time;
  for (size_t xs=0; xs<N; ++xs) {
    for(size_t js=0; js< l+1; ++js){
      printf("%f ", z[js + (l+1)*xs]);
    }
    printf("\n");
  }
#else 
  printf("Elapsed: %f\n", time);
#endif

  free(z);

}



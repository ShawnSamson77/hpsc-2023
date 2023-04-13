#include <cstdio>
#include <omp.h>
#include <time.h>

int main() {
  int n = 2e+9;
  double dx = 1. / n;
  double pi = 0;
  time_t start_time, end_time;

  start_time = time(NULL);
#pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  end_time = time(NULL);

  printf("%17.15f\n",pi);
  printf("calculation time: %ld sec\n",end_time - start_time);
}

//My laptop i7-1185G7 @3.00GHz, DDR4 16GB 4267MHz
//n = 2e+9, 1 thread  11 sec
//n = 2e+9, 8 threads  2 sec

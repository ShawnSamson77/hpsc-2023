#include <cstdio>
#include <omp.h>
#include <chrono>
#include <iostream>

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  int n = 2e+9;
  double dx = 1. / n;
  double pi = 0;

#pragma omp parallel for reduction(+:pi)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  
  printf("%17.15f\n",pi);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "calculate time: " << elapsed_time.count() << " seconds" << std::endl;
}

//My laptop i7-1185G7 @3.00GHz, DDR4 16GB 4267MHz
//n = 2e+9, 1 thread  2.04 sec
//n = 2e+9, 8 threads 0.56 sec

//g++ 12_pi.cpp -fopenmp -O3


#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();

  int n = 1e+9;
  int range = 100;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    //printf("%d ",key[i]);
  }
  //printf("\n");

  std::vector<int> bucket(range,0); 
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Single thread calculate time: " << elapsed_time.count() << " seconds" << std::endl;

  auto openMP_start_time = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for
  for (int i=0; i<n; i++)
//#pragma omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  end_time = std::chrono::high_resolution_clock::now();

  //for (int i=0; i<n; i++) {printf("%d ",key[i]);}
  //printf("\n");

  elapsed_time = end_time - openMP_start_time;
  std::chrono::duration<double> All_time = end_time - start_time;
  std::cout << "openMP calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  std::cout << "Total calculate time: " << All_time.count() << " seconds" << std::endl;

}

//Single thread part, n = 1e+9, 10.8 sec;  n = 2e+9, 22.3 sec
//n = 1e+9,  1 thread   11.8  sec
//n = 1e+9, 24 threads  11.5  sec

//n = 2e+9,  1 thread   24.2  sec
//n = 2e+9, 24 threads  23.5  sec




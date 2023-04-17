#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  int n = 50;
  int range = 200;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Single thread calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  auto openMP_start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
  for (int i=0; i<n; i++)
#pragma omp atomic update    //Single thread faster than omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  std::vector<int> temporal(range,0);
  
#pragma omp parallel for
  for (int i=0; i<range; i++)
    offset[i] = bucket[i];
  for(int j=1; j<range; j<<=1){
#pragma omp parallel for
    for(int i=0; i<range;i++){
      temporal[i] = offset[i];
    }
#pragma omp parallel for
    for(int i=j; i<range;i++){
      offset[i] += temporal[i-j];
    }
  }

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j;
    if (i > 0){j = offset[i-1];}
    else{j = 0;}
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  end_time = std::chrono::high_resolution_clock::now();

  for (int i=0; i<n; i++) {printf("%d ",key[i]);}
  printf("\n");

  elapsed_time = end_time - openMP_start_time;
  std::chrono::duration<double> All_time = end_time - start_time;
  std::cout << "openMP calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  std::cout << "Total calculate time: " << All_time.count() << " seconds" << std::endl;
}
//My laptop i7-1185G7 @3.00GHz, DDR4 16GB 4267MHz
//Single thread part, n = 1e+8, range = 1e+8, 0.91 sec
//n = 1e+8,  1 thread   5.39  sec
//n = 1e+8,  8 threads  4.01  sec
//g++ -fopenmp -O3 14_bucket_sort.cpp

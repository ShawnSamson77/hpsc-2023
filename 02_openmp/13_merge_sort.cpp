#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) { 
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++]; 
  }
  for (int i=0; i<tmp.size(); i++) 
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;
#pragma omp task shared(vec) firstprivate(begin,mid)
    merge_sort(vec, begin, mid);
#pragma omp task shared(vec) firstprivate(mid,end)
    merge_sort(vec, mid+1, end);
#pragma omp taskwait
    merge(vec, begin, mid, end);
  }
}

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  int n = 5e+7;
  std::vector<int> vec(n);

  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    //printf("%d ",vec[i]);
    }
  //printf("\n");

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Single thread calculate time: " << elapsed_time.count() << " seconds" << std::endl;

  auto openMP_start_time = std::chrono::high_resolution_clock::now();
  merge_sort(vec, 0, n-1);
  end_time = std::chrono::high_resolution_clock::now();

  //for (int i=0; i<n; i++) {printf("%d ",vec[i]);}
  //printf("\n");
  elapsed_time = end_time - openMP_start_time;
  std::chrono::duration<double> All_time = end_time - start_time;
  std::cout << "openMP calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  std::cout << "Total calculate time: " << All_time.count() << " seconds" << std::endl;

}

//Single thread part, n = 1e+8, 1.3 sec;  n = 2e+8, 2.2 sec;
//n = 1e+8,  1 thread   14.1  sec
//n = 1e+8, 24 threads  14.0  sec

//n = 2e+8,  1 thread   28.9  sec
//n = 2e+8, 24 threads  28.8  sec

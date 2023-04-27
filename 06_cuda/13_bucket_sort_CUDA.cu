#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <iostream>

__global__ void initialize(int *bucket, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  bucket[i] = 0;
}

__global__ void accumulate_bucket(int *bucket, int *key,int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void accumulate_key(int *bucket, int *key, int range, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int index = 0;
  for(int j=0; j<range;j++){
    index += bucket[j];
    if (i < index){
      key[i] = j;
      break;
    }
  }
}

int main() {
  auto start_time = std::chrono::high_resolution_clock::now();
  int n = 3000;
  int range = 3000;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Single thread calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  auto CUDA_start_time = std::chrono::high_resolution_clock::now();
  initialize<<<(range+1024-1)/1024, 1024>>>(bucket, range);
  cudaDeviceSynchronize();
  accumulate_bucket<<<(n+1024-1)/1024, 1024>>>(bucket, key, n);
  cudaDeviceSynchronize();
  accumulate_key<<<(n+1024-1)/1024, 1024>>>(bucket, key, range, n);
  cudaDeviceSynchronize();
  end_time = std::chrono::high_resolution_clock::now();

  for (int i=0; i<n; i++) {printf("%d ",key[i]);}
  printf("\n");

  elapsed_time = end_time - CUDA_start_time;
  std::chrono::duration<double> All_time = end_time - start_time;
  std::cout << "CUDA calculate time: " << elapsed_time.count() << " seconds" << std::endl;
  std::cout << "Total calculate time: " << All_time.count() << " seconds" << std::endl;

  cudaFree(key);
  cudaFree(bucket);
}
//n=2e+6, range=2e+6  3.05 sec

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float reduce(int N, __m256 avec){
  float a[N];
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(a, bvec);
  return a[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];  //add j[N]
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }

  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 jvec = _mm256_load_ps(j);
  __m256 mask, rxvec, ryvec, rvec, m3rvec, fxvec, fyvec;

  for(int i=0; i<N; i++) {
    mask = _mm256_cmp_ps(_mm256_set1_ps(i), jvec, _CMP_EQ_OQ);  //if i == j mask=1, else mask=0

    rxvec = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);  //rx = x[i] -x [j]
    ryvec = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);  //ry = y[i] - y[j]
    rvec = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));  //sqrt(rx*rx+ry*ry)

    m3rvec = _mm256_div_ps(mvec, _mm256_mul_ps(_mm256_mul_ps(rvec, rvec), rvec));  //  m / (r*r*r)
    fxvec = _mm256_blendv_ps(_mm256_mul_ps(m3rvec, rxvec), _mm256_set1_ps(0), mask);  //if mask==0 -> calculate, mask==1 -> f=0
    fyvec = _mm256_blendv_ps(_mm256_mul_ps(m3rvec, ryvec), _mm256_set1_ps(0), mask);

    fx[i] -= reduce(N, fxvec);
    fy[i] -= reduce(N, fyvec);
      
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

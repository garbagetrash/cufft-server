#include <math.h>
#include <stdio.h>

#include <cufft.h>

#include "cufft_server.h"

// CUDA kernel function to add the elements of two arrays on the GPU
__global__
// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

void c2c_fft_batch(float *h_data, int nfft, int batch)
{
    cufftHandle plan;
    cufftResult r = cufftPlan1d(&plan, nfft, CUFFT_C2C, batch);

    // Create device arrays
    cufftComplex *d_data = nullptr;
    cudaMalloc((void **)d_data, sizeof(cufftComplex) * nfft * batch);
    cudaMemcpy(d_data, h_data, 2 * sizeof(float) * nfft * batch, cudaMemcpyHostToDevice);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaMemcpy(h_data, d_data, 2 * sizeof(float) * nfft * batch, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    cufftDestroy(plan);
}

void add_example()
{
  int N = 1<<20; // 1M elements

  float *x;
  float *y;
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  }
  printf("Max error: %f\n", maxError);

  cudaFree(x);
  cudaFree(y);
}

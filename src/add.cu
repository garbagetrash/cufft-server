#include <complex>
#include <chrono>
#include <iostream>
#include <math.h>
#include <vector>

#include <cufft.h>

using namespace std::chrono;

// Some types
typedef std::complex<float> cf32;
typedef std::vector<cf32> vec_cf32;

// Prototypes
void c2c_fft(vec_cf32 &h_data, int nfft, int batch);

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

void test_fft()
{
    int nfft = 4096;
    int batch = 4096;
    vec_cf32 data(nfft * batch, 0);
    for (size_t i = 0; i < nfft * batch; i++) {
        data[i] = cf32(i, -i);
    }
    auto t1 = high_resolution_clock::now();
    c2c_fft(data, nfft, batch);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << std::endl;
}

void c2c_fft(vec_cf32 &h_data, int nfft, int batch)
{
    cufftHandle plan;
    cufftResult r = cufftPlan1d(&plan, nfft, CUFFT_C2C, batch);

    // Create device arrays
    cufftComplex *d_data = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(cufftComplex) * nfft * batch);
    cudaMemcpy(d_data, h_data.data(), sizeof(cf32) * h_data.size(), cudaMemcpyHostToDevice);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaMemcpy(h_data.data(), d_data, sizeof(cf32) * h_data.size(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_data);
    cufftDestroy(plan);
}

void add_example()
{
  int N = 1<<20; // 1M elements

  float *x, *y;
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
  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(x);
  cudaFree(y);
}

int main(void)
{
    add_example();

    // FFT EXAMPLE
    test_fft();

    return 0;
}

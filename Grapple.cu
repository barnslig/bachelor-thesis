#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#include "Grapple.h"

/**
 * Create a waypoints model successor
 *
 * @param process Idx of the process, 0...7
 * @param ndc Nondeterministic branch, 0...3
 * @param state Current state
 * @return The successor state
 */
__device__ uint32_t successor_generation(unsigned int process, unsigned int ndc, uint32_t state)
{
  return state | 1 << ((4 * process) + ndc);
}

/**
 * The Grapple CUDA kernel
 */
__global__ void Grapple(int *output)
{
  *output += 3;
}

int runGrapple()
{
  cudaError_t err = cudaSuccess;

  // Amount of parallel verification tests. Each corresponds to a CUDA block
  int VTs = 250;

  // Amount of threads in a verification test. Each corresponds to a CUDA thread in a CUDA block
  int N = 32;

  // Queue length per thread
  int I = 4;

  int threads_per_block = 1; // N;
  int blocks_per_grid = 1;   // VTs;

  // allocate device memory for our output
  int *output;
  cudaMalloc(&output, sizeof(int));

  // fill device memory with some value
  int host_output = 1;
  cudaMemcpy(output, &host_output, sizeof(int), cudaMemcpyHostToDevice);

  Grapple<<<threads_per_block, blocks_per_grid>>>(output);

  // copy device memory back into host memory
  cudaMemcpy(&host_output, output, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Output: " << host_output << "\n";

  // Check that the kernel launch was successful
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to launch kernel! Error code: " << cudaGetErrorString(err) << "\n";
    return EXIT_FAILURE;
  }

  err = cudaDeviceReset();
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to deinitialize the device! Error code: " << cudaGetErrorString(err) << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "done!\n";
  return EXIT_SUCCESS;
}

#include <iostream>

#include "Hashtable.cuh"
#include "State.cuh"

__global__ void HashtableKernel()
{
  __shared__ Hashtable table;
  // Wipe the hashtable memory on the first thread
  if (threadIdx.x == 0)
  {
    memset(&table, 0, sizeof(table));
  }
  __syncthreads();

  printf("From within the hashtable kernel!!\n");

  State state1 = {1};
  State state2 = {2};
  State state3 = {4};

  printf("2 is visited? %i\n", table.markVisited(&state1, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));
  printf("2 is visited? %i\n", table.markVisited(&state1, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));
  printf("3 is visited? %i\n", table.markVisited(&state2, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));
  printf("3 is visited? %i\n", table.markVisited(&state2, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));
  printf("4 is visited? %i\n", table.markVisited(&state3, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));
  printf("4 is visited? %i\n", table.markVisited(&state3, 0x9e3779b9, 0x9e3779b9, 0x9e3779b9));

  printf("Hashtable kernel end!\n");
}

int main()
{
  cudaError_t err = cudaSuccess;

  HashtableKernel<<<1, 1>>>();

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

  return 0;
}

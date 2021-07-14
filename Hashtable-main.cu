#include <iostream>

#include "Hashtable.cuh"
#include "State.cuh"

__global__ void HashtableKernel()
{
  __shared__ Hashtable table;
  if (threadIdx.x == 0)
  {
    hashtable_init(&table);
  }
  __syncthreads();

  printf("From within the hashtable kernel!!\n");

  State state = {12341};

  uint32_t state_hash = hash((uint8_t *)&state, sizeof(state), threadIdx.x) & hashmask(HASHTABLE_CAPACITY);

  /* Each hash bucket can store eight bits, each representing whether a
   * state is already visited or not. Thus, we have to divide the hash
   * by eight. This also saves us a lot of memory!
   * The modulo operation from the paper is omitted here as we mask
   * the hash so it does not exceed our hashtable.
   */
  uint32_t hashed_value = state_hash / 8;

  /* Determine which bit within our hash bucket represents the current
   * state by using modulo
   */
  uint32_t sel = hashed_value % 8;

  // Whether the current state is already visited
  bool is_visited = (table.elems[hashed_value] & (1 << sel)) != 0;

  // Set the current state as visited
  table.elems[hashed_value] |= (1 << sel);

  printf("%i, %i, %i, %i\n", state_hash, hashed_value, sel, is_visited);
  printf("Hashtable kernel end!\n");
}

int main()
{
  cudaError_t err = cudaSuccess;

  HashtableKernel<<<1, 3>>>();
  cudaDeviceSynchronize();

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

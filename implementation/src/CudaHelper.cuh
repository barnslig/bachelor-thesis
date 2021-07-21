#ifndef CUDA_HELPER_CUH_
#define CUDA_HELPER_CUH_

#include <cuda_runtime.h>

/**
 * A wrapper method to catch and handle errors caused by cuda* functions
 *
 * @example
 *   gpuErrchk(cudaMalloc(...));
 *
 * @see https://stackoverflow.com/a/14038590
 */
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

#endif // CUDA_HELPER_CUH_

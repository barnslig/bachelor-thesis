#ifndef GRAPPLE_CUH_
#define GRAPPLE_CUH_

#include "State.cuh"

// Amount of parallel verification tests. Each corresponds to a CUDA block
constexpr int kGrappleVTs = 250;

// Amount of threads in a verification test. Each corresponds to a CUDA thread in a CUDA block
constexpr int kGrappleN = 32;

/**
 * Run the Grapple model checker on the GPU
 *
 * @param runIdx Idx of the program execution
 * @param initialState The initial state
 * @param stream The CUDA stream in which we should execute the kernel
 * @returns 0 or an error code
 */
int runGrapple(int runIdx, State initialState, cudaStream_t *stream);

#endif // GRAPPLE_CUH_

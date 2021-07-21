#ifndef GRAPPLE_CUH_
#define GRAPPLE_CUH_

#include "State.cuh"

/**
 * Run the Grapple model checker on the GPU
 *
 * @param runIdx Idx of the program execution
 * @param initialState The initial state
 * @returns 0 or an error code
 */
int runGrapple(int runIdx, State initialState);

#endif // GRAPPLE_CUH_

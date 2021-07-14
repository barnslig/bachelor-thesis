#include "State.cuh"

__device__ State State::successor_generation(unsigned int process, unsigned int ndc)
{
  return State{state | 1 << ((4 * process) + ndc)};
}

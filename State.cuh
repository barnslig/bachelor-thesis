/**
 * @file
 * @brief A state successor generator
 */
#ifndef STATE_CUH_
#define STATE_CUH_

#include <cstdint>

/**
 * A current state of the waypoints model
 */
class State
{
  public:
  /**
   * The current waypoints model state
   */
  int32_t state;

  /**
   * Create a waypoints model successor
   *
   * @param process Idx of the process, 0...7
   * @param ndc Nondeterministic branch, 0...3
   * @returns The successor state
   */
  __device__ State successor_generation(unsigned int process, unsigned int ndc);

  /**
   * Get whether the state violates
   *
   * @returns Whether the state violates
   */
  __device__ bool violates();
};

#endif // STATE_CUH_

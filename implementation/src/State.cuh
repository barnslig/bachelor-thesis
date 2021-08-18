/**
 * @file
 * @brief A state successor generator
 */
#ifndef STATE_CUH_
#define STATE_CUH_

#include <cstdint>
#include <string>

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

  __host__ __device__ State() : state(0){};
  __host__ __device__ State(int32_t state) : state(state){};

  /**
   * Create a waypoints model successor
   *
   * @param successor A pointer where the successor is written to
   * @param process Idx of the process, 0...7
   * @param ndc Nondeterministic branch, 0...3
   */
  __host__ __device__ void successor_generation(State *successor, unsigned int process, unsigned int ndc);

  /**
   * Format the state as a string. Only used on the host
   *
   * @returns String representation of the state
   */
  __host__ std::string str();

  /**
   * Get whether the state violates
   *
   * @returns Whether the state violates
   */
  __host__ __device__ bool violates();
};

#endif // STATE_CUH_

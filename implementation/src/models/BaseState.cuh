/**
 * @file
 * @brief The abstract class that all models are based on
 */
#ifndef BASE_STATE_CUH_
#define BASE_STATE_CUH_

#include <string>

/**
 * Branching-Free Ternary Operator
 *
 * In CUDA, regular branching trough switch-/if-conditions, as usually
 * used for FSM implementation, causes threads to pause, resulting in
 * significant slow-down when heavily used.
 *
 * This method instead "calculates" all ternary operations, resulting
 * in good use of the SIMT execution model and good CUDA performance.
 *
 * See https://doi.org/10.1145/2632362.2632379
 *
 * @param cond The condition
 * @param valTrue The return value when `cond` evaluates to true
 * @param valFalse The return value when `cond` evaluates to false
 * @returns `valTrue` if `cond` evaluates to true, else `valFalse`
 */
__host__ __device__ inline uint8_t MyTernary(bool cond, uint8_t valTrue, uint8_t valFalse)
{
  return cond * valTrue + !cond * valFalse;
}

/**
 * The abstract class that all models are based on
 *
 * In order to be hashed by MurmurHash3_128, we have to align the model
 * onto 64-bit.
 *
 * @tparam T The class that inherits the BaseState, i.e. `MyState : public BaseState<MyState>`
 */
template <typename T>
class alignas(8) BaseState
{
  public:
  /**
   * Number of processes in the model
   */
  static const unsigned int kProcesses;

  /**
   * Number of nondeterministic choices on each state in the model
   */
  static const unsigned int kNondeterministicChoices;

  /**
   * The total state space size
   */
  static const unsigned int kStateSpaceSize;

  /**
   * Create a state successor
   *
   * @param successor The pointer to a state instance that the successor is written into
   * @param process The idx of the process, 0...kProcesses
   * @param ndc The nondeterministic branch, 0...kNondeterministicChoices
   */
  __host__ __device__ void successor_generation(T *successor, unsigned int process, unsigned int ndc);

  /**
   * Get whether the state violates
   *
   * @returns Whether the state violates
   */
  __host__ __device__ bool violates();

  /**
   * Format the state as a string. Only used on the host
   *
   * @returns String representation of the state
   */
  __host__ std::string str();
};

#endif // BASE_STATE_CUH_

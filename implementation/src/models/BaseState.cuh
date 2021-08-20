/**
 * @file
 * @brief The abstract class that all models are based on
 */
#ifndef BASE_STATE_CUH_
#define BASE_STATE_CUH_

#include <string>

/**
 * The abstract class that all models are based on
 *
 * @tparam T The class that inherits the BaseState, i.e. `MyState : public BaseState<MyState>`
 */
template <typename T>
class BaseState
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

#ifndef PHILOSOPHERS_STATE_CUH_
#define PHILOSOPHERS_STATE_CUH_

#include <cstring>

#include "BaseState.cuh"

/**
 * All possible states of each philosopher
 */
enum class PhilosophersStatesEnum
{
  kThink = 0,
  kOne = 1,
  kEat = 2,
  kFinish = 3
};

/**
 * A current state of the philosophers model
 */
class PhilosophersState : public BaseState<PhilosophersState>
{
  public:
  /**
   * The amount of philosophers (= processes) in the model
   */
  static const unsigned int kProcesses = 15;

  /**
   * The amount of nondeterministic choices per state. Always 1
   */
  static const unsigned int kNondeterministicChoices = 1;

  /**
   * The current state of each philosopher in the model
   */
  PhilosophersStatesEnum state[kProcesses];

  /**
   * The current state of each fork in the model
   *
   * 0 = resting
   * 1 = picked up
   */
  bool fork[kProcesses];

  __host__ __device__ PhilosophersState();

  __host__ __device__ PhilosophersState(const PhilosophersState &obj);

  __host__ __device__ void successor_generation(PhilosophersState *successor, unsigned int process, unsigned int ndc);

  __host__ __device__ bool violates();

  __host__ std::string str();
};

#endif // PHILOSOPHERS_STATE_CUH_

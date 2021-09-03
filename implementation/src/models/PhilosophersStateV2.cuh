/**
 * @file
 * @brief A model for the Dining Philosophers Problem
 */
#ifndef PHILOSOPHERS_STATE_V2_CUH_
#define PHILOSOPHERS_STATE_V2_CUH_

#include <sstream>

#include "BaseState.cuh"

/**
 * A current state of the Dining Philosophers Problem
 *
 * Based on phils.mdve from https://paradise.fi.muni.cz/beem/
 *
 * We search for one reachability property, namely the deadlock when
 * all philosophers have picked up waiting one fork, waiting for each
 * other to release the other.
 *
 * To achieve good performance with CUDA's SIMT execution model, we
 * calculate all expressions on every successor generation.
 * See https://doi.org/10.1145/2632362.2632379
 */
class PhilosophersStateV2 : public BaseState<PhilosophersStateV2>
{
  public:
  static const unsigned int kProcesses = 15;
  static const unsigned int kNondeterministicChoices = 1;
  static const unsigned int kStateSpaceSize = 14348906; // pow(3, kProcesses) - 1;

  bool fork[kProcesses] = {};

  /**
   * The current state of each process
   *
   * Possible values:
   *  0: think (initial state)
   *  1: one
   *  2: eat
   *  3: finish
   */
  uint8_t state[kProcesses] = {};

  __host__ __device__ PhilosophersStateV2();

  __host__ __device__ PhilosophersStateV2(const PhilosophersStateV2 &obj);

  __host__ __device__ void successor_generation(PhilosophersStateV2 *successor, unsigned int process, unsigned int ndc);

  __host__ __device__ bool violates();

  std::string str();
};

#endif // PHILOSOPHERS_STATE_V2_CUH_

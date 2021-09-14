/**
 * @file
 * @brief A model for the Peterson mutual exclusion protocol
 */
#ifndef PETERSON_STATE_CUH_
#define PETERSON_STATE_CUH_

#include <sstream>

#include "BaseState.cuh"

/**
 * A current state of the Peterson model
 *
 * Based on peterson.mdve from https://paradise.fi.muni.cz/beem/
 *
 * We search for one reachability property, namely the violation of
 * mutual exclusion when more than one process is in the critical
 * section.
 *
 * To achieve good performance with CUDA's SIMT execution model, we
 * calculate all expressions on every successor generation.
 * See https://doi.org/10.1145/2632362.2632379
 */
class PetersonState : public BaseState<PetersonState>
{
  public:
  static const unsigned int kProcesses = 5;
  static const unsigned int kNondeterministicChoices = 1;
  static const unsigned int kStateSpaceSize = UINT_MAX; // Unknown

  /**
   * The current state of each process
   *
   * Possible values:
   * 0: NCS (initial state)
   * 1: CS
   * 2: wait
   * 3: q2
   * 4: q3
   */
  uint8_t state[kProcesses] = {};

  uint8_t pos[kProcesses] = {};
  uint8_t step[kProcesses] = {};

  uint8_t j[kProcesses] = {};
  uint8_t k[kProcesses] = {};

  __host__ __device__ void successor_generation(PetersonState *successor, unsigned int process, unsigned int ndc);

  __host__ __device__ bool violates();

  std::string str();
};

#endif // PETERSON_STATE_CUH_

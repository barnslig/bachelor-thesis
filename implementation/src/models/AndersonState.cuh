/**
 * @file
 * @brief A model for the Andersons queue lock mutual exclusion algorithm
 * @see https://doi.org/10.1007/s00446-003-0088-6
 */
#ifndef ANDERSON_STATE_CUH_
#define ANDERSON_STATE_CUH_

#include <sstream>

#include "BaseState.cuh"

/**
 * A current state of the Anderson model
 *
 * Based on anderson.mdve from https://paradise.fi.muni.cz/beem/
 *
 * We search for one reachability property, namely the violation of
 * mutual exclusion when more than one process is in the critical
 * section.
 *
 * To achieve good performance with CUDA's SIMT execution model, we
 * calculate all expressions on every successor generation.
 * See https://doi.org/10.1145/2632362.2632379
 */
class AndersonState : public BaseState<AndersonState>
{
  public:
  static const unsigned int kProcesses = 3; // With more than 11, CUDA fails
  static const unsigned int kNondeterministicChoices = 1;
  static const unsigned int kStateSpaceSize = UINT_MAX; // Unknown

  bool slots[kProcesses] = {};
  uint8_t next = 0;
  uint8_t my_place[kProcesses] = {};

  /**
   * The current state of each process
   *
   * Possible values:
   *  0: NCS (initial state)
   *  1: p1
   *  2: p2
   *  3: p3
   *  4: CS
   */
  uint8_t state[kProcesses] = {};

  __host__ __device__ AndersonState();
  __host__ __device__ void successor_generation(AndersonState *successor, unsigned int process, unsigned int ndc);
  __host__ __device__ bool violates();
  std::string str();
};

#endif // ANDERSON_STATE_CUH_

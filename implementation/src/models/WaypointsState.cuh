/**
 * @file
 * @brief A state successor generator
 */
#ifndef WAYPOINTS_STATE_CUH_
#define WAYPOINTS_STATE_CUH_

#include <cstdint>

#include "BaseState.cuh"

/**
 * A current state of the waypoints model
 */
class WaypointsState : public BaseState<WaypointsState>
{
  private:
  /**
   * The current waypoints model state
   */
  int32_t state;

  public:
  static const unsigned int kProcesses = 8;
  static const unsigned int kNondeterministicChoices = 4;
  static const unsigned int kStateSpaceSize = 4294967295;

  __host__ __device__ WaypointsState() : state(0){};

  __host__ __device__ WaypointsState(const WaypointsState &obj) : state(obj.state){};

  __host__ __device__ void successor_generation(WaypointsState *successor, unsigned int process, unsigned int ndc);

  __host__ __device__ bool violates();

  __host__ std::string str();
};

#endif // WAYPOINTS_STATE_CUH_

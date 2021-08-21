#include <sstream>

#include "PhilosophersState.cuh"

__host__ __device__ PhilosophersState::PhilosophersState()
{
  for (size_t i = 0; i < kProcesses; i += 1)
  {
    state[i] = PhilosophersStatesEnum::kThink;
    fork[i] = false;
  }
}

__host__ __device__ PhilosophersState::PhilosophersState(const PhilosophersState &obj)
{
  memcpy(state, obj.state, kProcesses * sizeof(PhilosophersStatesEnum));
  memcpy(fork, obj.fork, kProcesses * sizeof(bool));
}

__host__ __device__ void PhilosophersState::successor_generation(PhilosophersState *successor, unsigned int process, unsigned int ndc)
{
  switch (state[process])
  {
  case PhilosophersStatesEnum::kThink:
  {
    if (fork[process] == false)
    {
      // pick up right fork
      successor->fork[process] = true;
      successor->state[process] = PhilosophersStatesEnum::kOne;
    }
    return;
  }

  case PhilosophersStatesEnum::kOne:
  {
    int nextFork = (process + 1) % kProcesses;
    if (fork[nextFork] == false)
    {
      // pick up left fork
      successor->fork[nextFork] = true;
      successor->state[process] = PhilosophersStatesEnum::kEat;
    }
    return;
  }

  case PhilosophersStatesEnum::kEat:
  {
    // put the right fork down
    successor->fork[process] = false;
    successor->state[process] = PhilosophersStatesEnum::kFinish;
    return;
  }

  case PhilosophersStatesEnum::kFinish:
  {
    // put the left fork down
    int nextFork = (process + 1) % kProcesses;
    successor->fork[nextFork] = false;
    successor->state[process] = PhilosophersStatesEnum::kThink;
    return;
  }
  }
}

__host__ __device__ bool PhilosophersState::violates()
{
  /**
   * Whether all philosophers are holding only the left fork
   */
  bool allHaveOne = true;
  for (int i = 0; i < kProcesses; i += 1)
  {
    if (state[i] != PhilosophersStatesEnum::kOne)
    {
      allHaveOne = false;
      break;
    }
  }

  /**
   * Whether all forks are picked up
   */
  bool allForksTaken = true;
  for (int i = 0; i < kProcesses; i += 1)
  {
    if (!fork[i])
    {
      allForksTaken = false;
      break;
    }
  }

  return allHaveOne && allForksTaken;
}

__host__ std::string PhilosophersState::str()
{
  std::ostringstream ss;
  for (int i = 0; i < kProcesses; i += 1)
  {
    ss << (int)state[i];
  }
  ss << "|";
  for (int i = 0; i < kProcesses; i += 1)
  {
    ss << (int)fork[i];
  }
  return ss.str();
}

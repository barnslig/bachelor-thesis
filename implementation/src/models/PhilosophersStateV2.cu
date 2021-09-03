#include "PhilosophersStateV2.cuh"

__host__ __device__ PhilosophersStateV2::PhilosophersStateV2() {}

__host__ __device__ PhilosophersStateV2::PhilosophersStateV2(const PhilosophersStateV2 &obj)
{
  memcpy(state, obj.state, sizeof(state));
  memcpy(fork, obj.fork, sizeof(fork));
}

__host__ __device__ void PhilosophersStateV2::successor_generation(PhilosophersStateV2 *successor, unsigned int process, unsigned int ndc)
{
  bool guard1 = state[process] == 0 && fork[process] == false;
  bool guard2 = state[process] == 1 && fork[(process + 1) % kProcesses] == false;
  bool guard3 = state[process] == 2;
  bool guard4 = state[process] == 3;

  // think -> one {guard fork[$2] == 0; effect fork[$2] = 1;},
  successor->fork[process] = MyTernary(guard1, true, fork[process]);
  successor->state[process] = MyTernary(guard1, 1, state[process]);

  // one -> eat {guard fork[$3] == 0; effect fork[$3] = 1;},
  successor->fork[(process + 1) % kProcesses] = MyTernary(guard2, true, fork[(process + 1) % kProcesses]);
  successor->state[process] = MyTernary(guard2, 2, successor->state[process]);

  // eat -> finish {effect fork[$2] = 0; },
  successor->fork[process] = MyTernary(guard3, false, successor->fork[process]);
  successor->state[process] = MyTernary(guard3, 3, successor->state[process]);

  // finish -> think {effect fork[$3] = 0; };
  successor->fork[(process + 1) % kProcesses] = MyTernary(guard4, false, successor->fork[(process + 1) % kProcesses]);
  successor->state[process] = MyTernary(guard4, 0, successor->state[process]);
}

__host__ __device__ bool PhilosophersStateV2::violates()
{
  /**
     * The number of philosophers in state "one"
     */
  unsigned int areOne = 0;

  /**
     * The number of picked-up forks
     */
  unsigned int forksTaken = 0;

  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    areOne += state[i] == 1;
    forksTaken += fork[i];
  }

  return areOne == kProcesses && forksTaken == kProcesses;
}

std::string PhilosophersStateV2::str()
{
  std::ostringstream ss;
  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    ss << std::to_string(state[i]);
  }
  ss << "|";
  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    ss << std::to_string(fork[i]);
  }
  return ss.str();
}

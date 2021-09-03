#include "AndersonState.cuh"

__host__ __device__ AndersonState::AndersonState()
{
  slots[0] = 1;
}

__host__ __device__ void AndersonState::successor_generation(AndersonState *successor, unsigned int process, unsigned int ndc)
{
  bool guard1 = state[process] == 0;
  bool guard2 = state[process] == 1 && my_place[process] == kProcesses - 1;
  bool guard3 = state[process] == 1 && my_place[process] != kProcesses - 1;
  bool guard4 = state[process] == 2 && slots[my_place[process]] == 1;
  bool guard5 = state[process] == 3;
  bool guard6 = state[process] == 4;

  // NCS -> p1 { effect my_place = next, next = next+1; },
  successor->my_place[process] = MyTernary(guard1, next, my_place[process]);
  successor->next = MyTernary(guard1, next + 1, next);
  successor->state[process] = MyTernary(guard1, 1, state[process]);

  // p1 -> p2 { guard my_place == N-1; effect next = next-N; },
  successor->next = MyTernary(guard2, next - kProcesses, successor->next);
  successor->state[process] = MyTernary(guard2, 2, successor->state[process]);

  // p1 -> p2 { guard my_place != N-1; effect my_place = my_place%N; },
  successor->my_place[process] = MyTernary(guard3, my_place[process] % kProcesses, successor->my_place[process]);
  successor->state[process] = MyTernary(guard3, 2, successor->state[process]);

  // p2 -> p3 { guard Slot[my_place] == 1;  },
  successor->state[process] = MyTernary(guard4, 3, successor->state[process]);

  // p3 -> CS { effect ifelse(ERROR,0, `Slot[my_place]=0', ERROR, 1, `Slot[(my_place+N-1)%N]=0'); },
  // We have no artificial error
  successor->slots[my_place[process]] = MyTernary(guard5, 0, slots[my_place[process]]);
  successor->state[process] = MyTernary(guard5, 4, successor->state[process]);

  // CS -> NCS { effect Slot[(my_place+1)%N]=1;};
  successor->slots[(my_place[process] + 1) % kProcesses] = MyTernary(guard6, 1, slots[(my_place[process] + 1) % kProcesses]);
  successor->state[process] = MyTernary(guard6, 0, successor->state[process]);
}

__host__ __device__ bool AndersonState::violates()
{
  /**
     * The number of processes in state "CS"
     */
  unsigned int inCS = 0;

  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    inCS += state[i] == 4;
  }

  return inCS > 1;
}

std::string AndersonState::str()
{
  std::ostringstream ss;

  ss << "slots: ";
  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    ss << std::to_string(slots[i]) << ", ";
  }

  ss << "next: ";
  ss << std::to_string(next) << ", ";

  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    ss << "process " << i << ": {";

    ss << "state: ";
    ss << std::to_string(state[i]) << ", ";

    ss << "my_place: ";
    ss << std::to_string(my_place[i]) << ", ";

    ss << "} ";
  }

  return ss.str();
}

#include "PetersonState.cuh"

__host__ __device__ void PetersonState::successor_generation(PetersonState *successor, unsigned int process, unsigned int ndc)
{
  bool guard1 = state[process] == 0;
  bool guard2 = state[process] == 2 && j[process] < kProcesses;
  bool guard3 = state[process] == 3;
  bool guard4 = state[process] == 4 && k[process] < kProcesses && (k[process] == 0 || pos[k[process]] < j[process]);
  bool guard5 = state[process] == 4 && step[j[process] - 1] != process || k[process] == kProcesses;
  bool guard6 = state[process] == 2 && j[process] == kProcesses;
  bool guard7 = state[process] == 1;

  // NCS -> wait { effect j = 1; },
  successor->j[process] = MyTernary(guard1, 1, j[process]);
  successor->state[process] = MyTernary(guard1, 2, state[process]);

  // wait -> q2  { guard j < N; effect pos[$1] = j;},
  successor->pos[process] = MyTernary(guard2, j[process], pos[process]);
  successor->state[process] = MyTernary(guard2, 3, successor->state[process]);

  // q2 -> q3 { effect step[j-1] = $1, k = 0; },
  successor->step[j[process] - 1] = MyTernary(guard3, process, step[j[process] - 1]);
  successor->k[process] = MyTernary(guard3, 0, k[process]);
  successor->state[process] = MyTernary(guard3, 4, successor->state[process]);

  // q3 -> q3 { guard k < N && (k == $1 || pos[k] ifelse(ERROR,1, `<=', `<') j); effect k = k+1;},
  // We have no artificial error
  successor->k[process] = MyTernary(guard4, k[process] + 1, successor->k[process]);
  successor->state[process] = MyTernary(guard4, 4, successor->state[process]);

  // q3 -> wait { guard ifelse(ERROR,2,`pos',`step')[j-1] != $1 || k == N; effect j = j+1;},
  successor->j[process] = MyTernary(guard5, j[process] + 1, successor->j[process]);
  successor->state[process] = MyTernary(guard5, 2, successor->state[process]);

  // wait -> CS { guard j == N; },
  successor->state[process] = MyTernary(guard6, 1, successor->state[process]);

  // CS -> NCS { effect pos[$1] = 0;};
  successor->pos[process] = MyTernary(guard7, 0, successor->pos[process]);
  successor->state[process] = MyTernary(guard7, 0, successor->state[process]);
}

__host__ __device__ bool PetersonState::violates()
{
  /**
   * The number of processes in state "CS"
   */
  unsigned int inCS = 0;

  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    inCS += state[i] == 1;
  }

  return inCS > 1;
}

std::string PetersonState::str()
{
  std::ostringstream ss;

  for (unsigned int i = 0; i < kProcesses; i += 1)
  {
    ss
        << std::to_string(state[i]) << "!"
        << std::to_string(pos[i]) << "!"
        << std::to_string(step[i]) << "!"
        << std::to_string(j[i]) << "!"
        << std::to_string(k[i]);

    if (i != kProcesses - 1)
    {
      ss << "|";
    }
  }

  return ss.str();
}

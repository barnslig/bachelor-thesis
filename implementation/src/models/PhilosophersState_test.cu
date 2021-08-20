#include <gtest/gtest.h>

#include <cstring>

#include "PhilosophersState.cuh"

using StateArr = PhilosophersStatesEnum[PhilosophersState::kProcesses];
using ForkArr = bool[PhilosophersState::kProcesses];

TEST(PhilosophersState, GeneratesSuccessors)
{
  PhilosophersState *next = new PhilosophersState;
  StateArr expectedState = {PhilosophersStatesEnum::kThink};
  ForkArr expectedFork = {false};

  // For each philosopher, test the whole think-one-eat-finish-think cycle
  for (int i = 0; i < PhilosophersState::kProcesses; i += 1)
  {
    EXPECT_EQ(memcmp(next->state, expectedState, sizeof(StateArr)), 0);
    EXPECT_EQ(memcmp(next->fork, expectedFork, sizeof(ForkArr)), 0);

    next = new PhilosophersState(*next);
    next->successor_generation(next, i, 0);
    expectedState[i] = PhilosophersStatesEnum::kOne;
    expectedFork[i] = true;
    EXPECT_EQ(memcmp(next->state, expectedState, sizeof(StateArr)), 0);
    EXPECT_EQ(memcmp(next->fork, expectedFork, sizeof(ForkArr)), 0);

    next = new PhilosophersState(*next);
    next->successor_generation(next, i, 0);
    expectedState[i] = PhilosophersStatesEnum::kEat;
    expectedFork[(i + 1) % PhilosophersState::kProcesses] = true;
    EXPECT_EQ(memcmp(next->state, expectedState, sizeof(StateArr)), 0);
    EXPECT_EQ(memcmp(next->fork, expectedFork, sizeof(ForkArr)), 0);

    next = new PhilosophersState(*next);
    next->successor_generation(next, i, 0);
    expectedState[i] = PhilosophersStatesEnum::kFinish;
    expectedFork[i] = false;
    EXPECT_EQ(memcmp(next->state, expectedState, sizeof(StateArr)), 0);
    EXPECT_EQ(memcmp(next->fork, expectedFork, sizeof(ForkArr)), 0);

    next = new PhilosophersState(*next);
    next->successor_generation(next, i, 0);
    expectedState[i] = PhilosophersStatesEnum::kThink;
    expectedFork[(i + 1) % PhilosophersState::kProcesses] = false;
    EXPECT_EQ(memcmp(next->state, expectedState, sizeof(StateArr)), 0);
    EXPECT_EQ(memcmp(next->fork, expectedFork, sizeof(ForkArr)), 0);
  }
}

TEST(PhilosophersState, FindsViolation)
{
  PhilosophersState *next = new PhilosophersState;

  /* A violation (= deadlock) is found when all philosophers have
   * picked up one fork and are waiting to pick up the second fork
   */
  for (int i = 0; i < PhilosophersState::kProcesses; i += 1)
  {
    next->state[i] = PhilosophersStatesEnum::kOne;
    next->fork[i] = true;
  }
  EXPECT_TRUE(next->violates());
}

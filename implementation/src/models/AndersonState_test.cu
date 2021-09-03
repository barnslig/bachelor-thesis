#include <gtest/gtest.h>
#include <iostream>

#include "AndersonState.cuh"

TEST(AndersonState, FindsViolation)
{
  AndersonState state;
  for (unsigned int i = 0; i < AndersonState::kProcesses; i += 1)
  {
    state.state[i] = 4;
  }

  EXPECT_TRUE(state.violates());
}

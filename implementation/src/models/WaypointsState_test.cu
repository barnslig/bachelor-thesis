#include <gtest/gtest.h>
#include <iostream>

#include "WaypointsState.cuh"

TEST(WaypointsState, GeneratesSuccessors)
{
  WaypointsState state;
  EXPECT_EQ(state.str(), std::to_string(0b00000000000000000000000000000000));

  WaypointsState successor(state);
  state.successor_generation(&successor, 0, 0);
  EXPECT_EQ(successor.str(), std::to_string(0b00000000000000000000000000000001));

  successor.successor_generation(&successor, 1, 0);
  EXPECT_EQ(successor.str(), std::to_string(0b00000000000000000000000000010001));

  successor.successor_generation(&successor, 2, 3);
  EXPECT_EQ(successor.str(), std::to_string(0b00000000000000000000100000010001));
}

#include <gtest/gtest.h>

#include "Hashtable.cuh"

TEST(HashtableTest, MarkVisited)
{
  Hashtable table = {};

  int a = 0x9e3779b9, b = 0x9e3779b9, c = 0x9e3779b9;

  State state1 = {1};
  State state2 = {2};
  State state3 = {4};

  ASSERT_FALSE(table.markVisited(&state1, a, b, c));
  ASSERT_TRUE(table.markVisited(&state1, a, b, c));
  ASSERT_FALSE(table.markVisited(&state2, a, b, c));
  ASSERT_TRUE(table.markVisited(&state2, a, b, c));
  ASSERT_FALSE(table.markVisited(&state3, a, b, c));
  ASSERT_TRUE(table.markVisited(&state3, a, b, c));
}

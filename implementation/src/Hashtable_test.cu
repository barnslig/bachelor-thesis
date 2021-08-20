#include <gtest/gtest.h>

#include "Hashtable.cuh"

using TestHashtable = Hashtable<int, 18>;

TEST(HashtableTest, MarkVisited)
{
  TestHashtable table = {};

  int a = 0x9e3779b9, b = 0x9e3779b9, c = 0x9e3779b9;

  int state1 = 1;
  int state2 = 2;
  int state3 = 4;

  ASSERT_FALSE(table.markVisited(&state1, a, b, c));
  ASSERT_TRUE(table.markVisited(&state1, a, b, c));
  ASSERT_FALSE(table.markVisited(&state2, a, b, c));
  ASSERT_TRUE(table.markVisited(&state2, a, b, c));
  ASSERT_FALSE(table.markVisited(&state3, a, b, c));
  ASSERT_TRUE(table.markVisited(&state3, a, b, c));
}

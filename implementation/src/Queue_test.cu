#include <gtest/gtest.h>

#include "Queue.cuh"

using TestQueue = Queue<int, 3>;

TEST(Queue, SimplePushAndPop)
{
  TestQueue q = {};

  EXPECT_TRUE(q.empty());
  q.push(1234);
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(*q.pop(), 1234);
  EXPECT_TRUE(q.empty());
}

TEST(Queue, FullQueueDropsNewElements)
{
  TestQueue q = {};

  EXPECT_TRUE(q.empty());
  q.push(1);
  q.push(2);
  q.push(3);
  q.push(4);
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(*q.pop(), 1);
  EXPECT_EQ(*q.pop(), 2);
  EXPECT_EQ(*q.pop(), 3);
  EXPECT_EQ(q.pop(), nullptr);
  EXPECT_TRUE(q.empty());
}

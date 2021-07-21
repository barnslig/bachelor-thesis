#include <gtest/gtest.h>

#include "Queue.cuh"

TEST(Queue, SimplePushAndPop)
{
  Queue q = {};

  EXPECT_TRUE(q.empty());
  q.push(State{10});
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(q.pop()->state, 10);
  EXPECT_TRUE(q.empty());
}

TEST(Queue, FullQueueDropsNewStates)
{
  Queue q = {};

  EXPECT_TRUE(q.empty());
  q.push(State{10});
  q.push(State{11});
  q.push(State{12});
  q.push(State{13});
  q.push(State{14});
  EXPECT_FALSE(q.empty());
  EXPECT_EQ(q.pop()->state, 10);
  EXPECT_EQ(q.pop()->state, 11);
  EXPECT_EQ(q.pop()->state, 12);
  EXPECT_EQ(q.pop()->state, 13);
  EXPECT_EQ(q.pop(), nullptr);
  EXPECT_TRUE(q.empty());
}

#include <gtest/gtest.h>

#include "CheapRingBuffer.cuh"

using TestCheapRingBuffer = CheapRingBuffer<int, 3>;

TEST(CheapRingBuffer, SimplePushAndPop)
{
  TestCheapRingBuffer buf = {};

  EXPECT_TRUE(buf.empty());
  buf.push(1234);
  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(*buf.pop(), 1234);
  EXPECT_TRUE(buf.empty());
}

TEST(CheapRingBuffer, FullBufferOverridesOldElements)
{
  TestCheapRingBuffer buf = {};

  EXPECT_TRUE(buf.empty());
  buf.push(1);
  buf.push(2);
  buf.push(3);
  buf.push(4);
  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(*buf.pop(), 4);
  EXPECT_EQ(*buf.pop(), 2);
  EXPECT_EQ(*buf.pop(), 3);
  EXPECT_EQ(buf.pop(), nullptr);
  EXPECT_TRUE(buf.empty());
}

TEST(CheapRingBuffer, BufferCanBeCleared)
{
  TestCheapRingBuffer buf = {};

  EXPECT_TRUE(buf.empty());
  buf.push(1);
  EXPECT_FALSE(buf.empty());
  buf.clear();
  EXPECT_TRUE(buf.empty());
}

#include <gtest/gtest.h>

#include "OutputBuffer.cuh"

using TestOutputBuffer = OutputBuffer<int, 3>;

TEST(OutputBuffer, SimplePushAndPop)
{
  TestOutputBuffer buf = {};

  EXPECT_TRUE(buf.empty());
  buf.push(1234);
  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(*buf.pop(), 1234);
  EXPECT_TRUE(buf.empty());
}

TEST(OutputBuffer, FullBufferDropsNewStates)
{
  TestOutputBuffer buf = {};

  EXPECT_TRUE(buf.empty());
  buf.push(1);
  buf.push(2);
  buf.push(3);
  buf.push(4);
  EXPECT_FALSE(buf.empty());
  EXPECT_EQ(*buf.pop(), 1);
  EXPECT_EQ(*buf.pop(), 2);
  EXPECT_EQ(*buf.pop(), 3);
  EXPECT_EQ(buf.pop(), nullptr);
  EXPECT_TRUE(buf.empty());
}

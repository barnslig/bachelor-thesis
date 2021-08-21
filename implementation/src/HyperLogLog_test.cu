#include <cstdio>
#include <gtest/gtest.h>

#include "HyperLogLog.cuh"

using TestCounter = HyperLogLog<int, sizeof(int), 10>;

TEST(HyperLogLogTest, AddAndEstimate)
{
  TestCounter counter;

  int a = 1234, b = 4321, c = 6374;
  counter.add(&a);
  counter.add(&a);
  counter.add(&b);
  counter.add(&c);

  std::cout << counter.estimate() << "\n";
}

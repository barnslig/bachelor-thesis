#include <cstdio>
#include <gtest/gtest.h>

#include "HyperLogLog.cuh"

using TestCounter = HyperLogLog<uint32_t, sizeof(uint32_t), 15>;

TEST(HyperLogLogTest, AddAndEstimate)
{
  TestCounter counter;

  unsigned int a = 1234, b = 4321, c = 6374;
  counter.add(&a);
  counter.add(&a);
  counter.add(&b);
  counter.add(&c);

  constexpr unsigned int kAdds = 100000;

  TestCounter counter2;
  for (uint32_t i = 0; i < kAdds; i += 1)
  {
    counter2.add(&i);
  }
  counter.merge(counter2);

  EXPECT_NEAR(counter.estimate(), kAdds + 3, counter.estimate() * counter.error());
}

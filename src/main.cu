#include "Grapple.cuh"

int main()
{
  // Each block is a VT and as we have 250 blocks per run, we just divide by this
  // TODO does this make sense? We still find way less waypoints than expected
  for (int i = 0; i < 10000 / 250; i += 1)
  {
    int ret = runGrapple(i, State{0});
    if (ret != 0)
    {
      return ret;
    }
  }
  return 0;
}

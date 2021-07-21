#include <cstdio>

#include "CudaHelper.cuh"
#include "Grapple.cuh"

/**
 * Number of Grapple runs
 *
 * Each block is a VT and as we have 250 (kGrappleVTs) blocks per run, we just
 * divide by this.
 */
// TODO does this make sense? We still find way less waypoints than expected
constexpr int kGrappleRuns = 50000 / kGrappleVTs;

int main()
{
  printf("run, block, thread, state\n");

  cudaStream_t stream[kGrappleRuns];
  int ret = 0;

  for (int i = 0; i < kGrappleRuns; i += 1)
  {
    /* Each Grapple run gets assigned to a different CUDA stream to achieve
     * maximum concurrency
     */
    cudaStreamCreate(&stream[i]);

    ret = runGrapple(i, State{0}, &stream[i]);
    if (ret != 0)
    {
      // Terminate program execution when a single Grapple run has failed
      goto terminate;
    }
  }

terminate:

  // Wait for all CUDA streams to terminate
  gpuErrchk(cudaDeviceSynchronize());

  // Check that the kernel launch was successful
  gpuErrchk(cudaGetLastError());

  // Final cleanup of the device before we leave
  gpuErrchk(cudaDeviceReset());

  return 0;
}

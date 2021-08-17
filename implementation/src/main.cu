#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <random>

#include "CudaHelper.cuh"
#include "Grapple.cuh"

/**
 * Default number of Grapple runs
 */
constexpr int kDefaultRuns = 50000;

int main(int argc, char *const argv[])
{
  std::random_device rd;

  int argRuns = kDefaultRuns;
  int argSeed = rd(); // 1736331306

  int c;
  while ((c = getopt(argc, argv, "s:n:h")) != -1)
  {
    switch (c)
    {
    case 's':
      argSeed = std::strtol(optarg, nullptr, 10);
      break;
    case 'n':
      /* Each block is a VT and as we have 250 (kGrappleVTs) blocks per
       * run, we just divide by this.
       */
      // TODO does this make sense? We still find way less waypoints than expected
      argRuns = std::strtol(optarg, nullptr, 10) / 250;
      break;
    case '?':
    case 'h':
      std::cerr << "Usage: " << argv[0] << " [options]\n\n";
      std::cerr << "Option        Description\n";
      std::cerr << " -s <seed>    Seed used for hash function diversification. Default: Random number\n";
      std::cerr << " -n <runs>    Number of Grapple runs. Default: " << kDefaultRuns << "\n";
      std::cerr << " -h           Show this message\n";
      exit(EXIT_FAILURE);
    }
  }

  printf("run, block, thread, state\n");

  std::mt19937 gen(argSeed);

  cudaStream_t stream[argRuns];
  int ret = 0;

  for (int i = 0; i < argRuns; i += 1)
  {
    /* Each Grapple run gets assigned to a different CUDA stream to achieve
     * maximum concurrency
     */
    cudaStreamCreate(&stream[i]);

    ret = runGrapple(i, State{0}, &gen, &stream[i]);
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

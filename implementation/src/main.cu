#include <unistd.h>

#include <iostream>
#include <random>
#include <unordered_set>

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

  std::cout << "run, block, thread, state, uniques, visited, visited_percent, vts\n";

  std::mt19937 gen(argSeed);

  cudaStream_t stream[argRuns];

  // Output of the last grapple run
  std::shared_ptr<GrappleOutput> out;

  // Set of all discovered violations. Used to track number of unique violations
  std::unordered_set<std::string> unique_violations;

  // HyperLogLog counter into which we merge all other counters
  StateCounter global_visited;

  for (unsigned int i = 0; i < argRuns; i += 1)
  {
    /* Each Grapple run gets assigned to a different CUDA stream to achieve
     * maximum concurrency
     */
    cudaStreamCreate(&stream[i]);

    out = runGrapple(i, &gen, &stream[i]);

    global_visited.merge(*out->visited.get());
    double estimate = global_visited.estimate();
    double visitedPercent = estimate / State::kStateSpaceSize * 100;
    std::cout
        << ",,,,,"
        << estimate
        << ","
        << visitedPercent
        << ","
        << (i + 1) * 250
        << "\n";

    while (!out->violations->empty())
    {
      Violation *v = out->violations->pop();
      unique_violations.insert(v->state.str());
      std::cout
          << v->run
          << ", "
          << v->block
          << ", "
          << v->thread
          << ", "
          << v->state.str()
          << ", "
          << unique_violations.size()
          << ",,,\n";
    }
  }

  // Wait for all CUDA streams to terminate
  gpuErrchk(cudaDeviceSynchronize());

  // Check that the kernel launch was successful
  gpuErrchk(cudaGetLastError());

  // Final cleanup of the device before we leave
  gpuErrchk(cudaDeviceReset());

  return 0;
}

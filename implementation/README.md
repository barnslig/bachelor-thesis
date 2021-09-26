# Experimental Grapple Model Checker

An experimental Swarm Verification model checker that leverages the GPU.

The implementation is part of my bachelor's thesis on _Low-Connectivity State Space Exploration using Swarm Model Checking on the GPU_ and based on [DeFrancisco, R. et al. "Swarm model checking on the GPU"](https://doi.org/10.1007/s10009-020-00576-x).

In particular, this model checker currently implements:

- Multiple models, including the **Waypoints Benchmark Model**. See [`src/models/`](src/models/README.md) for an overview of all available models.
- A **bitstate hashtable** that uses a single bit of each 32 bit hash bucket to mark a state as visited. Hashing is done using a modified variant of [Jenkins, Bob "A Hash Function for Hash Table Lookup"](http://www.burtleburtle.net/bob/hash/doobs.html).
- The parallel **breadth-first-search** (BFS) algorithm and its corresponding **queues** from [Holzmann, G. J. "Parallelizing the Spin Model Checker"](https://doi.org/10.1007/978-3-642-31759-0_12), as defined in the Swarm Model Checking paper.
- An extension to the search strategy called _start overs_ that can reach deeper states. See my thesis for details about it.

Only safety and reachability properties can be checked.

Please check out section 5.2 (Usage) of my thesis for a user guide.

## Quick Start

Make sure to have the prerequisites installed:

- [CUDA Toolkit](https://docs.nvidia.com/cuda/index.html#installation-guides)
- [cmake](https://cmake.org/)

Build the project using cmake:

```
cmake -B build
cmake --build build
```

The model checker can then be started using:

```
./build/grapple
```

## Build Configuration

The build can be configured using text macros.
Please check out section 5.2.2 (Build Configuration) of my thesis for a list of available text macros.

To specify configuration parameters, you can use an environment variable:

```
CUDA_FLAGS="-DGRAPPLE_MODEL=PhilosophersStateV2" cmake --build build -- -e
```

## Development

### Testing

Unit tests are written using [GoogleTest](https://google.github.io/googletest/).

Make sure the project is built before executing unit tests. Then, execute:

```
cd build
ctest
```

### Code formatting and linting

Code is formatted using [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html).

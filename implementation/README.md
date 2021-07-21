# Experimental Grapple Model Checker

An experimental Swarm Verification model checker that leverages the GPU.

The implementation is part of my bachelor's thesis on _Low-Connectivity State Space Exploration using Swarm Model Checking on the GPU_ and based on [DeFrancisco, R. et al. Swarm model checking on the GPU](https://doi.org/10.1007/s10009-020-00576-x).

In particular, this model checker currently implements:

- A single model called the **Waypoints Model**, as defined in [Holzmann, G. J. et al. Swarm Verification Techniques](https://doi.org/10.1109/TSE.2010.110). The current implementation leaves room to implement other models.
- A **bitstate hashtable** that uses a single bit of each 32 bit hash bucket to mark a state as visited. It is based on a modified variant of [Jenkins, Bob A Hash Function for Hash Table Lookup](http://www.burtleburtle.net/bob/hash/doobs.html).
- The parallel **breadth-first-search** algorithm and its corresponding **queues** from [Holzmann, G. J. Parallelizing the Spin Model Checker](https://doi.org/10.1007/978-3-642-31759-0_12), as defined in the Swarm Model Checking paper.

Only safety and reachability properties can be checked.

## Usage

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

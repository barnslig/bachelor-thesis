#ifndef GRAPPLE_CUH_
#define GRAPPLE_CUH_

#include <memory>

#include "CheapRingBuffer.cuh"
#include "Hashtable.cuh"
#include "HyperLogLog.cuh"
#include "OutputBuffer.cuh"
#include "Queue.cuh"
#include "models/PhilosophersState.cuh"
#include "models/WaypointsState.cuh"

// Amount of parallel verification tests. Each corresponds to a CUDA block
constexpr int kGrappleVTs = 250;

// Amount of threads in a verification test. Each corresponds to a CUDA thread in a CUDA block
constexpr int kGrappleN = 32;

/**
 * Times a VT may start over the search
 *
 * A "start over" happens when the VT is "done", i.e. the hash table
 * is full and every new state seems to be already visited. Then,
 * the hash table is cleared and the last set of visited, non-violating
 * states is used to start a new search within the same VT.
 *
 * To keep the algorithm terminating, this constant defines how often
 * it may start over.
 */
constexpr unsigned int kStartOvers = 10;

// Capacity of the violations output buffer, i.e. the maximum number of violations a single VT can report
constexpr unsigned int kViolationsOutputBufferSize = 256;

/**
 * The amount of states that can be marked in a hash table, as a power of two
 * 2^13*32/8000 = 32.768 kilobyte
 */
constexpr unsigned int kHashtableCapacity = 14;

/**
 * A single queue's capacity
 *
 * Implemented as a `constexpr` instead of a template class parameter so we do
 * not have to deal with non-specialized template compilation problems.
 * See https://stackoverflow.com/a/10632266
 */
constexpr unsigned int kQueueCapacity = 4;

/**
 * The size of the HyperLogLog register, as the power of two, i.e. 2^10
 */
constexpr unsigned int kHyperLogLogRegisters = 14;

using State = WaypointsState; // PhilosophersState;
using StateHashtable = Hashtable<State, kHashtableCapacity>;
using StateQueue = Queue<State, kQueueCapacity>;
using StateCounter = HyperLogLog<State, sizeof(State), kHyperLogLogRegisters>;
using StateRingBuffer = CheapRingBuffer<State *, kQueueCapacity>;

constexpr size_t kQueuesWidth = kGrappleVTs * 2 * kGrappleN * kGrappleN;
constexpr size_t kQueuesVTWidth = 2 * kGrappleN * kGrappleN;
constexpr size_t kQueuesPhaseWidth = kGrappleN * kGrappleN;
constexpr size_t kQueuesQueuesWidth = kGrappleN;

/**
 * The 1D size of Grapple queues
 *
 * The queue is of size kGrappleVTs x 2 x kGrappleN x kGrappleN, but we only
 * allocate a 1D array and calculate the addresses using `qAddr`.
 */
// TODO check alignment: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
constexpr size_t kQueuesSize = kQueuesWidth * sizeof(StateQueue);

constexpr size_t kHashPrimersWidth = kGrappleVTs * 3;
constexpr size_t kHashPrimersSize = kHashPrimersWidth * sizeof(int);

/**
 * A violation, found by a Grapple VT
 */
struct Violation
{
  /**
   * Idx of the run
   */
  unsigned int run;

  /**
   * Idx of the block, i.e. blockIdx.x
   */
  unsigned int block;

  /**
   * Idx of the thread, i.e. threadIdx.x
   */
  unsigned int thread;

  /**
   * The violating state
   */
  State state;
};

/**
 * The output buffer used to transfer violations from device to host
 */
using ViolationOutputBuffer = OutputBuffer<Violation, kViolationsOutputBufferSize>;

/**
 * The output generated by a Grapple run
 */
struct GrappleOutput
{
  /**
   * Discovered violations
   */
  std::shared_ptr<ViolationOutputBuffer> violations;

  /**
   * HyperLogLog visited states counter
   */
  std::shared_ptr<StateCounter> visited;
};

/**
 * Run the Grapple model checker on the GPU
 *
 * @param runIdx Idx of the program execution
 * @param distrib A uniform distribution to draw hash function parameters from
 * @param stream The CUDA stream in which we should execute the kernel
 * @returns The output of the VTs
 */
std::shared_ptr<GrappleOutput>
runGrapple(unsigned int runIdx, std::mt19937 *gen, cudaStream_t *stream);

#endif // GRAPPLE_CUH_

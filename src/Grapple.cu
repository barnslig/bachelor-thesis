#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include "CudaHelper.cuh"
#include "Grapple.cuh"
#include "Hashtable.cuh"
#include "Queue.cuh"
#include "State.cuh"

constexpr int kQueuesWidth = kGrappleVTs * 2 * kGrappleN * kGrappleN;
constexpr int kQueuesVTWidth = 2 * kGrappleN * kGrappleN;
constexpr int kQueuesPhaseWidth = kGrappleN * kGrappleN;
constexpr int kQueuesQueuesWidth = kGrappleN;

/* The 1D size of Grapple queues
 *
 * The queue is of size kGrappleVTs x 2 x kGrappleN x kGrappleN, but we only
 * allocate a 1D array and calculate the addresses using `qAddr`.
 */
// TODO check alignment: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
constexpr size_t kQueuesSize = kQueuesWidth * sizeof(Queue);

/**
 * Map the 4D queue array indices onto a 1D array
 *
 * The desired indexing is something like queue[0...249][0...1][0...31][0...31].
 * However, using multidimensional arrays in CUDA global memory is not easy. As
 * of this, we use a 1D array and map the indices onto it.
 *
 * @param vt Idx of the thread (= VT) within a block
 * @param t Current algorithm phase, either 0 or 1
 * @param x Output thread idx
 * @param y Input thread idx
 * @returns 1D queue array idx
 */
__device__ inline int qAddr(int vt, int t, int x, int y)
{
  return vt * kQueuesVTWidth + t * kQueuesPhaseWidth + x * kQueuesQueuesWidth + y;
}

/**
 * An array of three random hash function seeds for each VT (= one block)
 *
 * All random integers are stored in one big array placed in constant memory
 * on the device. To access them, use d_hash_primers[blockIdx.x],
 * d_hash_primers[blockIdx.x + 1] and d_hash_primers[blockIdx.x + 2].
 */
__constant__ int d_hash_primers[kGrappleVTs * 3];

/**
 * Get whether a VT is finished
 *
 * Definition of done: All "output queues", i.e. the queues used to fetch
 * states during the next algorithm phase, have to be empty.
 *
 * @param queue A pointer to the 4D queue array
 * @param t A pointer to the current algorithm phase
 * @returns Whether the next phase has no work to do
 */
__device__ bool check_done(Queue *queue, int *t)
{
  bool done = true;

  for (size_t i = 0; i < kGrappleN; i += 1)
  {
    for (size_t j = 0; j < kGrappleN; j += 1)
    {
      if (!queue[qAddr(blockIdx.x, (1 - *t), i, j)].empty())
      {
        done = false;
        goto done;
      }
    }
  }

done:
  return done;
}

/**
 * The Grapple CUDA kernel
 *
 * @param runIdx Idx of the program execution
 * @param queue A pointer to the multidimensional queues
 * @param initialState The initial state
 */
__global__ void Grapple(int runIdx, Queue *queue, State initialState)
{
  // The hashtable which tracks already visited states
  __shared__ Hashtable table;

  // The queue we are working on right now
  __shared__ int t;

  // The "rounds", i.e. toggles of t, that this block has done during execution
  __shared__ int rounds;

  // Initialize shared variables and initial state within the first thread
  if (threadIdx.x == 0)
  {
    memset(&table, 0, sizeof(table));
    t = 0;
    rounds = 0;
    queue[qAddr(blockIdx.x, t, threadIdx.x, 0)].push(initialState);
  }

  // Sync all threads after initial variable setup
  __syncthreads();

  bool done = false;
  while (!done)
  {
    // The idx of the next thread which gets new states pushed into its queue
    int next_output_i = 1;

    // For each input queue of this thread
    for (int i = 0; i < kGrappleN; i += 1)
    {
      // For each state in the input queue, until the queue is empty
      Queue *q = &queue[qAddr(blockIdx.x, t, threadIdx.x, i)];
      while (!q->empty())
      {
        // Pop the state from the queue
        State *s = q->pop();

        // printf("Thread %i, State %i, addr: %i\n", threadIdx.x, s->state, qAddr(blockIdx.x, t, threadIdx.x, i));

        // For each process in the model
        for (unsigned int p = 0; p < 8; p += 1)
        {
          // For each nondeterministic choice in the process
          for (unsigned int ndc = 0; ndc < 4; ndc += 1)
          {
            // Generate a successor of the state from the input queue
            State successor = s->successor_generation(p, ndc);
            bool is_visited = table.markVisited(&successor, d_hash_primers[blockIdx.x], d_hash_primers[blockIdx.x + 1], d_hash_primers[blockIdx.x + 2]);

            // printf("Thread %i: State %i visited: %i\n", threadIdx.x, successor.state, is_visited);

            // By now, we only care about unvisited states
            if (!is_visited)
            {
              if (successor.violates())
              {
                // When finding a violation (= a waypoint), only report it
                printf("%i, %i, %i, %i\n", runIdx, blockIdx.x, threadIdx.x, successor.state);
              }
              else
              {
                // TODO pick random output queue i. By now we dont care and cycle through a counter
                int this_i = next_output_i++ % kGrappleN;

                // printf("Thread %i adds state %i to thread %i.\n", threadIdx.x, successor.state, this_i);

                Queue *out = &queue[qAddr(blockIdx.x, (1 - t), this_i, threadIdx.x)];
                out->push(successor);
              }
            }
          }
        }
      }
    }

    // Sync all threads before swapping queues
    __syncthreads();

    done = check_done(queue, &t);

    // Swap queues
    if (threadIdx.x == 0)
    {
      rounds += 1;
      t = 1 - t;
    }
  }

  // printf("VT %i: Thread %i done.\n", blockIdx.x, threadIdx.x);

  if (threadIdx.x == 0)
  {
    // Calculate hashtable utilisation
    int used_buckets = 0;
    int used_slots = 0;
    const int table_size = hashsize(kHashtableCapacity) / 32;
    for (int i = 0; i < table_size; i += 1)
    {
      if (table.elems[i] != 0)
      {
        used_buckets += 1;
        used_slots += __popc(table.elems[i]);
      }
    }

    // Show block algorithm metrics
    // printf("%i: rounds %i, used buckets: %i, used slots: %i\n", blockIdx.x, rounds, used_buckets, used_slots);
  }
}

int runGrapple(int runIdx, State initialState, cudaStream_t *stream)
{
  int threads_per_block = kGrappleN;
  int blocks_per_grid = kGrappleVTs;

  // Allocate global device memory for queues
  Queue *d_queue;
  gpuErrchk(cudaMallocAsync(&d_queue, kQueuesSize, *stream));
  gpuErrchk(cudaMemsetAsync(d_queue, 0, kQueuesSize, *stream));

  /* Create three random integers for each block (= VT) as hash function seeds
   * See https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
   */
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(INT32_MIN, INT32_MAX);
  int h_hash_primers[kGrappleVTs * 3] = {};
  for (int i = 0; i < kGrappleVTs * 3; i += 1)
  {
    h_hash_primers[i] = distrib(gen);
  }
  gpuErrchk(cudaMemcpyToSymbolAsync(d_hash_primers, h_hash_primers, sizeof(h_hash_primers), 0, cudaMemcpyHostToDevice, *stream));

  // Run the kernel
  Grapple<<<blocks_per_grid, threads_per_block, 0, *stream>>>(runIdx, d_queue, initialState);

  // Free global memory allocated by `cudaMalloc`
  gpuErrchk(cudaFreeAsync(d_queue, *stream));

  return EXIT_SUCCESS;
}

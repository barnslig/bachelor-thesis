#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

#include "CudaHelper.cuh"
#include "Grapple.cuh"

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
__device__ inline size_t qAddr(size_t vt, size_t t, size_t x, size_t y)
{
  return vt * kQueuesVTWidth + t * kQueuesPhaseWidth + x * kQueuesQueuesWidth + y;
}

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
__device__ bool check_done(StateQueue *queue, int *t)
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
 * @param hashPrimers A pointer to an array containing three hash primers per block
 * @param counter A HyperLogLog counter in which the visited states are counted
 * @param output The violations output buffer
 */
__global__ void
Grapple(unsigned int runIdx, StateQueue *queue, int *hashPrimers, StateCounter *counter, ViolationOutputBuffer *output)
{
  // The hashtable which tracks already visited states
  __shared__ StateHashtable table;

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
    queue[qAddr(blockIdx.x, t, threadIdx.x, 0)].push(State{});
  }

  // Sync all threads after initial variable setup
  __syncthreads();

  int a = hashPrimers[blockIdx.x];
  int b = hashPrimers[blockIdx.x + 1];
  int c = hashPrimers[blockIdx.x + 2];

  bool done = false;
  while (!done)
  {
    // The idx of the next thread which gets new states pushed into its queue
    int next_output_i = 0;

    // For each input queue of this thread
    for (int i = 0; i < kGrappleN; i += 1)
    {
      // For each state in the input queue, until the queue is empty
      StateQueue *q = &queue[qAddr(blockIdx.x, t, threadIdx.x, i)];
      while (!q->empty())
      {
        // Pop the state from the queue
        State *s = q->pop();

        // printf("Thread %i, State %i, addr: %i\n", threadIdx.x, s->state, qAddr(blockIdx.x, t, threadIdx.x, i));

        // For each process in the model
        for (unsigned int p = 0; p < State::kProcesses; p += 1)
        {
          // For each nondeterministic choice in the process
          for (unsigned int ndc = 0; ndc < State::kNondeterministicChoices; ndc += 1)
          {
            /* Generate a successor of the state from the input queue
             *
             * We call the copy constructor to inherit the current
             * state. This is important as e.g. the dining philosophers
             * model only changes a few bits on the successor.
             */
            State successor(*s);
            s->successor_generation(&successor, p, ndc);
            bool is_visited = table.markVisited(&successor, a, b, c);

            // printf("Thread %i: State %i visited: %i\n", threadIdx.x, successor.state, is_visited);

            // By now, we only care about unvisited states
            if (!is_visited)
            {
              counter->add(&successor);

              if (successor.violates())
              {
                // When finding a violation (= a waypoint), report it
                Violation v = {
                    .run = runIdx,
                    .block = blockIdx.x,
                    .thread = threadIdx.x,
                    .state = successor,
                };
                output->push(v);
              }
              else
              {
                // TODO pick next random output queue i. By now we dont care and cycle through a counter
                next_output_i = (next_output_i + 1) % kGrappleN;

                // printf("Thread %i adds state %i to thread %i.\n", threadIdx.x, successor.state, next_output_i);

                StateQueue *out = &queue[qAddr(blockIdx.x, (1 - t), next_output_i, threadIdx.x)];
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

    __syncthreads();
  }

  // printf("VT %i: Thread %i done.\n", blockIdx.x, threadIdx.x);
}

std::shared_ptr<GrappleOutput>
runGrapple(unsigned int runIdx, std::mt19937 *gen, cudaStream_t *stream)
{
  int threads_per_block = kGrappleN;
  int blocks_per_grid = kGrappleVTs;

  // Allocate global device memory for queues
  StateQueue *d_queue;
  gpuErrchk(cudaMallocAsync(&d_queue, kQueuesSize, *stream));
  gpuErrchk(cudaMemsetAsync(d_queue, 0, kQueuesSize, *stream));

  /**
   * An array of three random hash function seeds for each VT (= one block)
   *
   * All random integers are stored in one big array placed in constant memory
   * on the device. To access them, use d_hash_primers[blockIdx.x],
   * d_hash_primers[blockIdx.x + 1] and d_hash_primers[blockIdx.x + 2].
   */
  int *d_hash_primers;
  gpuErrchk(cudaMallocAsync(&d_hash_primers, kHashPrimersSize, *stream));

  /* Create three random integers for each block (= VT) as hash function seeds
   * See https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
   */
  std::uniform_int_distribution<> distrib(INT32_MIN, INT32_MAX);
  int h_hash_primers[kHashPrimersWidth] = {};
  for (int i = 0; i < kHashPrimersWidth; i += 1)
  {
    h_hash_primers[i] = distrib(*gen);
  }
  gpuErrchk(cudaMemcpyAsync(d_hash_primers, h_hash_primers, kHashPrimersSize, cudaMemcpyHostToDevice, *stream));

  // Create a HyperLogLog counter for visited states
  StateCounter h_counter;
  StateCounter *d_counter;
  gpuErrchk(cudaMallocAsync(&d_counter, sizeof(StateCounter), *stream));
  gpuErrchk(cudaMemcpyAsync(d_counter, &h_counter, sizeof(StateCounter), cudaMemcpyHostToDevice, *stream));

  // Create an output buffer for discovered violations
  ViolationOutputBuffer *d_output;
  gpuErrchk(cudaMallocAsync(&d_output, sizeof(ViolationOutputBuffer), *stream));
  gpuErrchk(cudaMemsetAsync(d_output, 0, sizeof(ViolationOutputBuffer), *stream));

  // Run the kernel
  Grapple<<<blocks_per_grid, threads_per_block, 0, *stream>>>(runIdx, d_queue, d_hash_primers, d_counter, d_output);

  // Copy HyperLogLog counter of visited states back to host
  gpuErrchk(cudaMemcpyAsync(&h_counter, d_counter, sizeof(StateCounter), cudaMemcpyDeviceToHost, *stream));

  // Copy discovered violations back to host
  ViolationOutputBuffer h_output;
  gpuErrchk(cudaMemcpyAsync(&h_output, d_output, sizeof(ViolationOutputBuffer), cudaMemcpyDeviceToHost, *stream));

  // Free global memory allocated by `cudaMalloc`
  gpuErrchk(cudaFreeAsync(d_queue, *stream));
  gpuErrchk(cudaFreeAsync(d_hash_primers, *stream));
  gpuErrchk(cudaFreeAsync(d_counter, *stream));
  gpuErrchk(cudaFreeAsync(d_output, *stream));

  // Create and return the output struct
  GrappleOutput out = {
      .violations = std::make_shared<ViolationOutputBuffer>(h_output),
      .visited = std::make_shared<StateCounter>(h_counter),
  };
  return std::make_shared<GrappleOutput>(out);
}

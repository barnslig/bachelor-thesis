/**
 * @file
 * @brief A fixed-capacity fifo buffer, to be used as CUDA kernel output buffer
 */
#ifndef OUTPUT_BUFFER_CUH_
#define OUTPUT_BUFFER_CUH_

#include <cstdio>
#include <iostream>

#include "CudaHelper.cuh"

/**
 * A fixed-capacity fifo buffer, to be used as CUDA kernel output buffer
 *
 * The `OutputBuffer` is needed as our CUDA kernel produces an
 * unpredictable amount of outputs: A VT may yield anything between 0
 * and (number of total states) violations.
 *
 * By using a fixed-capacity elements array and array indices instead
 * of pointers for head and tail, we can copy an `OutputBuffer` between
 * host and device without worry.
 *
 * The buffer is designed to ONLY be used by:
 *  1) Push elements into the buffer
 *  2) Remove elements from the buffer
 *  3) Drop the buffer
 * Do NOT mix push and pop! To reuse a buffer, ALWAYS `memset` its
 * memory to 0!
 *
 * @tparam T The type of the elements hold by the buffer
 * @tparam N The buffer's capacity
 */
template <typename T, unsigned int N>
class OutputBuffer
{
  private:
  /**
   * Fixed-capacity array of buffer elements
   */
  T elems[N] = {};

  /**
   * Idx of the first element
   */
  unsigned int head = 0;

  /**
   * Idx of the last element
   */
  unsigned int tail = 0;

  public:
  /**
   * Insert an element at the back
   *
   * @param el The element to push into the buffer
   */
  __host__ __device__ void push(T el)
  {
    if (tail == N)
    {
      printf("Buffer full\n");
      return;
    }

    // The atomic increment saves us from thread synchronization issues
    unsigned int next = myAtomicInc(&tail, N);
    elems[next] = el;
  }

  /**
   * Remove and return the first element and increment
   *
   * @returns A pointer to the first element in the buffer or nullptr if empty
   */
  __host__ __device__ T *pop()
  {
    if (head == tail)
    {
      return nullptr;
    }
    return &elems[head++];
  }

  /**
   * Get whether the buffer is empty
   *
   * @returns Whether the buffer is empty
   */
  __host__ __device__ bool empty()
  {
    return head == tail;
  }
};

#endif // OUTPUT_BUFFER_CUH_

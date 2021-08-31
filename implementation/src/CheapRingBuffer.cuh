/**
 * @file
 * @brief A fixed-capacity buffer that overrides the oldest entry when full
 */
#ifndef CHEAP_RING_BUFFER_CUH_
#define CHEAP_RING_BUFFER_CUH_

/**
 * A fixed-capacity buffer that overrides the oldest entry when full
 *
 * This data structure is designed for the "start over" extension of
 * the Grapple state space exploration loop.
 *
 * It is *not* a full-fledged fifo ring buffer, as it does not
 * necessarily return its elements in fifo order. Also, it is designed
 * to be used in two phases: In the first phase, the buffer is filled.
 * In the second phase, it is consumed until empty. Phases should NOT
 * be mixed!
 *
 * @tparam T The type of the elements hold by the buffer
 * @tparam N The buffer's capacity
 */
template <typename T, unsigned int N>
class CheapRingBuffer
{
  private:
  /**
   * Fixed-capacity array of buffered elements
   */
  T elems[N] = {};

  /**
   * Idx of the first element
   */
  unsigned int head = 0;

  /**
   * Idx of the next element
   */
  unsigned int tail = 0;

  /**
   * Number of currently stored elements
   */
  unsigned int numElems = 0;

  public:
  /**
   * Insert an element
   *
   * When the buffer is full, this operation starts overriding the
   * oldest entry.
   *
   * @param el The element to push into the buffer
   */
  __host__ __device__ void push(T el)
  {
    numElems = min(N, numElems + 1);

    unsigned int old = tail;
    tail = (tail + 1) % N;
    elems[old] = el;
  }

  /**
   * Remove and return the first element and increment
   *
   * The first element is NOT necessarily the oldest!
   *
   * @returns A pointer to the first element in the buffer or nullptr if empty
   */
  __host__ __device__ T *pop()
  {
    if (numElems == 0)
    {
      return nullptr;
    }

    numElems -= 1;

    unsigned int old = head;
    head = (head + 1) % N;
    return &elems[old];
  }

  /**
   * Get whether the buffer is empty
   *
   * @returns Whether the buffer is empty
   */
  __host__ __device__ bool empty()
  {
    return numElems == 0;
  }

  /**
   * Clear the queue: Removes all entries
   */
  __host__ __device__ void clear()
  {
    memset(elems, 0, N);
    head = 0;
    tail = 0;
    numElems = 0;
  }
};

#endif // CHEAP_RING_BUFFER_CUH_

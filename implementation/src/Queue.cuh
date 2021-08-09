/**
 * @file
 * @brief A fixed-capacity fifo queue, designed for G. J. Holzmann's parallel BFS
 */
#ifndef QUEUE_CUH_
#define QUEUE_CUH_

/**
 * The queue's capacity
 *
 * Implemented as a `constexpr` instead of a template class parameter so we do
 * not have to deal with non-specialized template compilation problems.
 * See https://stackoverflow.com/a/10632266
 */
constexpr int kQueueCapacity = 4;

#include "State.cuh"

/**
 * A queue element of the fixed-capacity fifo queue
 */
class QueueElem
{
  public:
  /**
   * The state which is stored within this queue element
   */
  State state;

  /**
   * The successor of this queue element. `nullptr` if there is none
   */
  QueueElem *next; // = nullptr;
};

/**
 * A fixed-capacity fifo queue, designed for G. J. Holzmann's parallel BFS
 *
 * The queue is not a true circular buffer as it does not track start
 * and end idx. It only works under the constraints of the two-phase
 * BFS algorithm from G. J. Holzmann's "Parallelizing the Spin Model
 * Checker" paper: During the first phase, the queue is filled. During
 * the second phase, the queue is emptied again. As of this, we do not
 * have to keep track of our position within the memory.
 */
class Queue
{
  private:
  /**
   * The statically allocated queue elems
   */
  QueueElem elems[kQueueCapacity]; // = {};

  /**
   * The pointer to the first element in the queue
   */
  QueueElem *head; // = nullptr;

  /**
   * The pointer to the last element in the queue
   */
  QueueElem *tail; // = nullptr;

  public:
  __host__ __device__ Queue() : elems(), head(nullptr), tail(nullptr){};

  /**
   * Insert an element at the back
   *
   * New elements are silently dropped when the queue is full.
   *
   * @param state The state to push into the queue
   */
  __host__ __device__ void push(State state);

  /**
   * Remove and return the first element
   *
   * Returning a pointer to the state is sufficient as the queue's `elems`
   * stay untouched until the BFS algorithm's next round.
   *
   * @returns A pointer to the state which is removed from the queue or null
   */
  __host__ __device__ State *pop();

  /**
   * Get whether a queue is empty
   *
   * @returns Whether the queue is empty
   */
  __host__ __device__ bool empty() const;
};

#endif // QUEUE_CUH_

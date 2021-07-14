/**
 * @file
 * @brief A fixed-capacity fifo queue, designed for G. J. Holzmann's parallel BFS
 */
#ifndef __QUEUE_H_
#define __QUEUE_H_

#ifndef QUEUE_CAPACITY
#define QUEUE_CAPACITY 4
#endif

#include "State.cuh"

/**
 * Data structure for a queue element of the fixed-capacity fifo queue
 */
struct QueueElem
{
  // The state which is stored within this queue element
  State state;

  // The successor of this queue element. `nullptr` if there is none
  QueueElem *next;
};

/**
 * Data structure for a fixed-capacity fifo queue
 *
 * The queue is not a true circular buffer as it does not track start
 * and end idx. It only works under the constraints of the two-phase
 * BFS algorithm from G. J. Holzmann's "Parallelizing the Spin Model
 * Checker" paper: During the first phase, the queue is filled. During
 * the second phase, the queue is emptied again. As of this, we do not
 * have to keep track of our position within the memory.
 */
struct Queue
{
  // The statically allocated queue elems
  QueueElem elems[QUEUE_CAPACITY];

  // The pointer to the first element in the queue
  QueueElem *head;

  // The pointer to the last element in the queue
  QueueElem *tail;
};

/**
 * Initialize a queue by setting everything to 0
 */
__device__ void queue_init(Queue *q);

/**
 * Insert an element at the back
 *
 * New elements are silently dropped when the queue is full.
 *
 * @param q The queue
 * @param state The state to push into the queue
 */
__device__ void queue_push(Queue *q, State state);

/**
 * Remove and return the first element
 *
 * @param q The queue
 * @returns A pointer to the state which is removed from the queue or null
 */
__device__ State *queue_pop(Queue *q);

/**
 * Get whether a queue is empty
 *
 * @returns Whether the queue is empty
 */
__device__ bool queue_empty(Queue *q);

#endif

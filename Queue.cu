#include <cstring>

#include "Queue.cuh"

__device__ void queue_init(Queue *q)
{
  memset(q, 0, sizeof(Queue));
}

__device__ void queue_push(Queue *q, State state)
{
  if (q->tail == &q->elems[QUEUE_CAPACITY - 1])
  {
    // Drop new elements when the queue is full
    return;
  }

  /* On an empty queue, the next elem is the first memory address of
   * `elems`. Otherwise, it is the tail's pointer + 1. Overflows are
   * already cleared by the condition above
   */
  QueueElem *nextElem = q->tail ? q->tail + 1 : q->elems;

  nextElem->state = state;
  nextElem->next = nullptr;

  // When the queue is empty, we also have to set the head
  if (!q->head)
  {
    q->head = nextElem;
  }

  // When the queue is empty, we also have to set the tail
  if (!q->tail)
  {
    q->tail = nextElem;
  }

  /* When the queue is not empty, we have to update the tail and the current
   * tail's successor
   */
  else
  {
    q->tail->next = nextElem;
    q->tail = nextElem;
  }
}

__device__ State *queue_pop(Queue *q)
{
  if (!q->head)
  {
    // Do nothing when the queue is empty
    return nullptr;
  }

  // Unset the tail when we pop the last element
  if (q->tail == q->head)
  {
    q->tail = nullptr;
  }

  /* Returning a pointer to the state is sufficient as the queue's
   * elems stay untouched until the algorithm's next round.
   */
  State *state = &q->head->state;

  q->head = q->head->next;

  return state;
}

__device__ bool queue_empty(Queue *q)
{
  return !q->head && !q->tail;
}

#include <cstring>

#include "Queue.cuh"

__device__ void Queue::push(State state)
{
  if (tail == &elems[QUEUE_CAPACITY - 1])
  {
    // Drop new elements when the queue is full
    return;
  }

  /* On an empty queue, the next elem is the first memory address of
   * `elems`. Otherwise, it is the tail's pointer + 1. Overflows are
   * already cleared by the condition above
   */
  QueueElem *nextElem = tail ? tail + 1 : elems;

  nextElem->state = state;
  nextElem->next = nullptr;

  // When the queue is empty, we also have to set the head
  if (!head)
  {
    head = nextElem;
  }

  // When the queue is empty, we also have to set the tail
  if (!tail)
  {
    tail = nextElem;
  }

  /* When the queue is not empty, we have to update the tail and the current
   * tail's successor
   */
  else
  {
    tail->next = nextElem;
    tail = nextElem;
  }
}

__device__ State *Queue::pop()
{
  if (!head)
  {
    // Do nothing when the queue is empty
    return nullptr;
  }

  // Unset the tail when we pop the last element
  if (tail == head)
  {
    tail = nullptr;
  }

  State *state = &head->state;

  head = head->next;

  return state;
}

__device__ bool Queue::empty() const
{
  return !head && !tail;
}

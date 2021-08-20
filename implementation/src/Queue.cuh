/**
 * @file
 * @brief A fixed-capacity fifo queue, designed for G. J. Holzmann's parallel BFS
 */
#ifndef QUEUE_CUH_
#define QUEUE_CUH_

/**
 * A queue element of the fixed-capacity fifo queue
 *
 * @tparam T The type of the element hold by the queue elem
 */
template <typename T>
class QueueElem
{
  public:
  /**
   * The queue element
   */
  T elem;

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
 *
 * @tparam T The type of the elements hold by the queue
 * @tparam N The queue's capacity
 */
template <typename T, unsigned int N>
class Queue
{
  private:
  /**
   * The statically allocated queue elems
   */
  QueueElem<T> elems[N]; // = {};

  /**
   * The pointer to the first element in the queue
   */
  QueueElem<T> *head; // = nullptr;

  /**
   * The pointer to the last element in the queue
   */
  QueueElem<T> *tail; // = nullptr;

  public:
  __host__ __device__ Queue() : elems(), head(nullptr), tail(nullptr){};

  /**
   * Insert an element at the back
   *
   * New elements are silently dropped when the queue is full.
   *
   * @param elem The element to push into the queue
   */
  __host__ __device__ void push(T elem)
  {
    if (tail == &elems[N - 1])
    {
      // Drop new elements when the queue is full
      return;
    }

    /* On an empty queue, the next elem is the first memory address of
   * `elems`. Otherwise, it is the tail's pointer + 1. Overflows are
   * already cleared by the condition above
   */
    QueueElem<T> *nextElem = tail ? tail + 1 : elems;

    nextElem->elem = elem;
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

  /**
   * Remove and return the first element
   *
   * Returning a pointer to the element is sufficient as the queue's `elems`
   * stay untouched until the BFS algorithm's next round.
   *
   * @returns A pointer to the element which is removed from the queue or nullptr
   */
  __host__ __device__ T *pop()
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

    T *elem = &head->elem;

    head = head->next;

    return elem;
  }

  /**
   * Get whether a queue is empty
   *
   * @returns Whether the queue is empty
   */
  __host__ __device__ bool empty() const
  {
    return !head && !tail;
  }
};

#endif // QUEUE_CUH_

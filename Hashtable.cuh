/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 * @see https://burtleburtle.net/bob/hash/doobs.html
 */
#ifndef __HASHTABLE_H_
#define __HASHTABLE_H_

#include "State.cuh"
#include <cstdint>

#ifndef HASHTABLE_CAPACITY
/**
 * The hashtable size, as a power of two
 *
 * 2^13*32/8000 = 32.768 kilobyte
 */
#define HASHTABLE_CAPACITY 18
#endif

#define hashsize(n) ((uint32_t)1 << (n))
#define hashmask(n) (hashsize(n) - 1)

/**
 * A hashtable for the Grapple model checker, based on the Jenkins hash functions
 */
class Hashtable
{
  public:
  /**
   * The hashtable's buckets
   *
   * We can divide by 32 as we do so as well on the hashes as each bucket gets
   * assigned eight bit of different information.
   *
   * Public so we can inspect the hashtable utilization when the algorithm
   * has finished.
   */
  uint32_t elems[hashsize(HASHTABLE_CAPACITY) / 32]; // = {};

  /**
   * Mark a state as visited
   *
   * @param state The state to mark as visited
   * @param a A random value to seed the hash
   * @param b A random value to seed the hash
   * @param c A random value to seed the hash
   * @returns Whether the state was previously already visited
   */
  __device__ bool markVisited(State *state, int a, int b, int c);
};

#endif

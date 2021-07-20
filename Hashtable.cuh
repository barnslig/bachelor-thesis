/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 * @see https://burtleburtle.net/bob/hash/doobs.html
 */
#ifndef HASHTABLE_CUH_
#define HASHTABLE_CUH_

#include <cstdint>

#include "State.cuh"

/**
 * The amount of states that can be marked in the table, as a power of two
 * 2^13*32/8000 = 32.768 kilobyte
 */
constexpr int kHashtableCapacity = 18;

__host__ __device__ constexpr uint32_t hashsize(uint32_t n)
{
  return (uint32_t)1 << n;
}

__host__ __device__ constexpr uint32_t hashmask(uint32_t n)
{
  return hashsize(n) - 1;
}

/**
 * A hashtable for the Grapple model checker, based on the Jenkins hash functions
 */
class Hashtable
{
  private:
  /**
   * The amount of buckets within the hashtable
   *
   * We can divide by 32 as we do so as well on the hashes and each bucket gets
   * assigned eight bit of different information.
   */
  static constexpr uint32_t kHashtableSize = hashsize(kHashtableCapacity) / 32;

  public:
  /**
   * The hashtable's buckets
   *
   * Public so we can inspect the hashtable utilization after the algorithm
   * has finished.
   */
  uint32_t elems[kHashtableSize]; // = {};

  /**
   * Mark a state as visited
   *
   * @param state The state to mark as visited
   * @param a A random value to seed the hash
   * @param b A random value to seed the hash
   * @param c A random value to seed the hash
   * @returns Whether the state was previously already visited
   */
  __host__ __device__ bool markVisited(State *state, int a, int b, int c);
};

#endif // HASHTABLE_CUH_

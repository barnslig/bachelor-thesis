/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 */
#ifndef HASHTABLE_CUH_
#define HASHTABLE_CUH_

#include <cstdint>

#include "CudaHelper.cuh"
#include "JenkinsHash.cuh"

/**
 * A hashtable for the Grapple model checker, based on the Jenkins hash functions
 *
 * @tparam T The type of the elements marked in the hash table
 * @tparam N The hashtable's capacity
 */
template <typename T, unsigned int N>
class Hashtable
{
  private:
  /**
   * The amount of buckets within the hashtable
   *
   * We can divide by 32 as we do so as well on the hashes and each bucket gets
   * assigned 32 bit of different information.
   */
  static constexpr uint32_t kHashtableSize = hashsize(N) / 32;

  public:
  /**
   * The hashtable's buckets
   *
   * Public so we can inspect the hashtable utilization after the algorithm
   * has finished.
   */
  uint32_t elems[kHashtableSize]; // = {};

  /**
   * Mark an element as visited
   *
   * @param elem The element to mark as visited
   * @param a A random value to seed the hash
   * @param b A random value to seed the hash
   * @param c A random value to seed the hash
   * @returns Whether the element was previously already visited
   */
  __host__ __device__ bool markVisited(T *elem, int a, int b, int c)
  {
    uint32_t elem_hash = jenkins_hash(reinterpret_cast<uint8_t *>(elem), sizeof(T), a, b, c) & hashmask(N);

    /* Each hash bucket can store 32 bits, each representing whether an
     * element is already visited or not. Thus, we have to divide the
     * hash by 32. This also saves us a lot of memory!
     * The modulo operation from the paper is omitted here as we mask
     * the hash so it does not exceed our hashtable.
     */
    uint32_t hashed_value = elem_hash / 32;

    /* Determine which bit within our hash bucket represents the current
     * element by using modulo
     */
    uint32_t sel = elem_hash % 32;

    bool is_visited = (myAtomicOr(&elems[hashed_value], (1 << sel)) & (1 << sel)) != 0;

    return is_visited;
  }
};

#endif // HASHTABLE_CUH_

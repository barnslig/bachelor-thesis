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
   *
   * For N = 18, we thus have:
   *  2^18 / 32
   *  = (1 << 18) / 32
   *  = (1 << 18 - 5)
   *  = (1 << 13)
   *  = 8192 buckets
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
    uint32_t elem_hash = jenkins_hash(elem, sizeof(T), a, b, c) & hashmask(N);

    /* Last N-5 bits: Bucket index
     *
     * In the paper, this calculation is done using: elem_hash / 32
     */
    uint32_t bucket_idx = elem_hash & (1 << N - 5) - 1;

    /* First 5 bits: Index inside the bucket
     *
     * Each hash bucket can store 32 bits, each representing whether an
     * element is already visited or not. This saves us a lot of memory!
     *
     * In the paper, this calculation is done using: elem_hash % 32
     */
    uint32_t elem_idx = (elem_hash >> N - 5);

    bool is_visited = (myAtomicOr(&elems[bucket_idx], (1 << elem_idx)) & (1 << elem_idx)) != 0;

    return is_visited;
  }
};

#endif // HASHTABLE_CUH_

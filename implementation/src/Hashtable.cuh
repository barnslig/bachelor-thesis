/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 * @see https://burtleburtle.net/bob/hash/doobs.html
 */
#ifndef HASHTABLE_CUH_
#define HASHTABLE_CUH_

#include <cstdint>

#include "CudaHelper.cuh"

__host__ __device__ constexpr uint32_t hashsize(uint32_t n)
{
  return (uint32_t)1 << n;
}

__host__ __device__ constexpr uint32_t hashmask(uint32_t n)
{
  return hashsize(n) - 1;
}

/**
 * Mix three 32-bit values reversibly
 *
 * For every delta with one or two bits set, and the deltas of all three
 * high bits or all three low bits, whether the original value of a,b,c
 * is almost all zero or is uniformly distributed,
 *
 * - If mix() is run forward or backward, at least 32 bits in a,b,c
 *   have at least 1/4 probability of changing.
 * - If mix() is run forward, every bit of c will change between 1/3 and
 *   2/3 of the time.  (Well, 22/100 and 78/100 for some 2-bit deltas.)
 *
 * mix() was built out of 36 single-cycle latency instructions in a
 * structure that could supported 2x parallelism, like so:
 *      a -= b;
 *      a -= c; x = (c>>13);
 *      b -= c; a ^= x;
 *      b -= a; x = (a<<8);
 *      c -= a; b ^= x;
 *      c -= b; x = (b>>13);
 *      ...
 * Unfortunately, superscalar Pentiums and Sparcs can't take advantage
 * of that parallelism.  They've also turned some of those single-cycle
 * latency instructions into multi-cycle latency instructions.  Still,
 * this is the fastest good hash I could find.  There were about 2^^68
 * to choose from. I only looked at a billion or so.
 *
 * @param a Any 4-byte value
 * @param b Any 4-byte value
 * @param c Any 4-byte value
 */
__host__ __device__ static inline void mix(uint32_t *a, uint32_t *b, uint32_t *c)
{
  *a -= *b;
  *a -= *c;
  *a ^= (*c >> 13);
  *b -= *c;
  *b -= *a;
  *b ^= (*a << 8);
  *c -= *a;
  *c -= *b;
  *c ^= (*b >> 13);
  *a -= *b;
  *a -= *c;
  *a ^= (*c >> 12);
  *b -= *c;
  *b -= *a;
  *b ^= (*a << 16);
  *c -= *a;
  *c -= *b;
  *c ^= (*b >> 5);
  *a -= *b;
  *a -= *c;
  *a ^= (*c >> 3);
  *b -= *c;
  *b -= *a;
  *b ^= (*a << 10);
  *c -= *a;
  *c -= *b;
  *c ^= (*b >> 15);
}

/**
 * Hash a variable-length key into a 32-bit value
 *
 * Returns a 32-bit value.  Every bit of the key affects every bit of
 * the return value.  Every 1-bit and 2-bit delta achieves avalanche.
 * About 6*len+35 instructions.
 *
 * The best hash table sizes are powers of 2. There is no need to do
 * mod a prime (mod is sooo slow!).  If you need less than 32 bits,
 * use a bitmask.  For example, if you need only 10 bits, do
 *   h = (h & hashmask(10));
 * In which case, the hash table should have hashsize(10) elements.
 *
 * If you are hashing n strings (uint8_t **)k, do it like this:
 *   for (i=0, h=0; i<n; ++i) h = hash( k[i], len[i], h);
 *
 * By Bob Jenkins, 1996.  bob_jenkins@burtleburtle.net.  You may use this
 * code any way you wish, private, educational, or commercial.  It's free.
 *
 * See http://burtleburtle.net/bob/hash/evahash.html
 * Use for hash table lookup, or anything where one collision in 2^^32 is
 * acceptable.  Do NOT use for cryptographic purposes.
 *
 * @param k The key (the unaligned variable-length array of bytes)
 * @param length The length of the key, counting by bytes
 * @param a Any 4-byte value
 * @param b Any 4-byte value
 * @param c Any 4-byte value
 * @returns The hash
 */
__host__ __device__ static inline uint32_t jenkins_hash(uint8_t *k, uint32_t length, uint32_t a, uint32_t b, uint32_t c)
{
  /* Set up the internal state */
  uint32_t len = length;

  /*---------------------------------------- handle most of the key */
  while (len >= 12)
  {
    a += (k[0] + ((uint32_t)k[1] << 8) + ((uint32_t)k[2] << 16) + ((uint32_t)k[3] << 24));
    b += (k[4] + ((uint32_t)k[5] << 8) + ((uint32_t)k[6] << 16) + ((uint32_t)k[7] << 24));
    c += (k[8] + ((uint32_t)k[9] << 8) + ((uint32_t)k[10] << 16) + ((uint32_t)k[11] << 24));
    mix(&a, &b, &c);
    k += 12;
    len -= 12;
  }

  /*------------------------------------- handle the last 11 bytes */
  c += length;
  switch (len) /* all the case statements fall through */
  {
  case 11:
    c += ((uint32_t)k[10] << 24);
  case 10:
    c += ((uint32_t)k[9] << 16);
  case 9:
    c += ((uint32_t)k[8] << 8);
    /* the first byte of c is reserved for the length */
  case 8:
    b += ((uint32_t)k[7] << 24);
  case 7:
    b += ((uint32_t)k[6] << 16);
  case 6:
    b += ((uint32_t)k[5] << 8);
  case 5:
    b += k[4];
  case 4:
    a += ((uint32_t)k[3] << 24);
  case 3:
    a += ((uint32_t)k[2] << 16);
  case 2:
    a += ((uint32_t)k[1] << 8);
  case 1:
    a += k[0];
    /* case 0: nothing left to add */
  }
  mix(&a, &b, &c);
  /*-------------------------------------------- report the result */
  return c;
}

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

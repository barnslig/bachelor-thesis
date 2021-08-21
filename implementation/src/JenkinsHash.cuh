/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 * @see https://burtleburtle.net/bob/hash/doobs.html
 */
#ifndef JENKINS_HASH_CUH_
#define JENKINS_HASH_CUH_

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
__host__ __device__ inline void mix(uint32_t *a, uint32_t *b, uint32_t *c);

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
__host__ __device__ uint32_t jenkins_hash(uint8_t *k, uint32_t length, uint32_t a, uint32_t b, uint32_t c);

#endif // JENKINS_HASH_CUH_

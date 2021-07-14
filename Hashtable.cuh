/**
 * @file
 * @brief A hashtable for the Grapple model checker, based on the Jenkins hash functions
 * @see https://burtleburtle.net/bob/hash/doobs.html
 */
#ifndef __HASHTABLE_H_
#define __HASHTABLE_H_

#ifndef HASHTABLE_CAPACITY
// The hashtable size, as a power of two
// 2^15*8/8000 = 32.768 kilobyte
#define HASHTABLE_CAPACITY 15
#endif

#define hashsize(n) ((uint32_t)1 << (n))
#define hashmask(n) (hashsize(n) - 1)

#include <cstdint>

struct Hashtable
{
  /**
   * The hashtable's buckets
   *
   * We can divide by 8 as we do so as well on the hashes as each bucket gets
   * assigned eight bit of different information.
   */
  uint8_t elems[hashsize(HASHTABLE_CAPACITY) / 8];
};

/**
 * Initialize a hashtable by setting everything to 0
 */
__device__ void hashtable_init(Hashtable *table);

/**
 * Mix three 32-bit values reversibly
 *
 * For every delta with one or two bits set, and the deltas of all three
 *   high bits or all three low bits, whether the original value of a,b,c
 *   is almost all zero or is uniformly distributed,
 * * If mix() is run forward or backward, at least 32 bits in a,b,c
 *   have at least 1/4 probability of changing.
 * * If mix() is run forward, every bit of c will change between 1/3 and
 *   2/3 of the time.  (Well, 22/100 and 78/100 for some 2-bit deltas.)
 * mix() was built out of 36 single-cycle latency instructions in a
 *   structure that could supported 2x parallelism, like so:
 *      a -= b;
 *      a -= c; x = (c>>13);
 *      b -= c; a ^= x;
 *      b -= a; x = (a<<8);
 *      c -= a; b ^= x;
 *      c -= b; x = (b>>13);
 *      ...
 *  Unfortunately, superscalar Pentiums and Sparcs can't take advantage
 *  of that parallelism.  They've also turned some of those single-cycle
 *  latency instructions into multi-cycle latency instructions.  Still,
 *  this is the fastest good hash I could find.  There were about 2^^68
 *  to choose from. I only looked at a billion or so.
*/
#define mix(a, b, c) \
  {                  \
    a -= b;          \
    a -= c;          \
    a ^= (c >> 13);  \
    b -= c;          \
    b -= a;          \
    b ^= (a << 8);   \
    c -= a;          \
    c -= b;          \
    c ^= (b >> 13);  \
    a -= b;          \
    a -= c;          \
    a ^= (c >> 12);  \
    b -= c;          \
    b -= a;          \
    b ^= (a << 16);  \
    c -= a;          \
    c -= b;          \
    c ^= (b >> 5);   \
    a -= b;          \
    a -= c;          \
    a ^= (c >> 3);   \
    b -= c;          \
    b -= a;          \
    b ^= (a << 10);  \
    c -= a;          \
    c -= b;          \
    c ^= (b >> 15);  \
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
 * @param len The length of the key, counting by bytes
 * @param initval Any 4-byte value
 * @returns The hash
 */
__device__ uint32_t hash(uint8_t *k, uint32_t length, uint32_t initval);

#endif

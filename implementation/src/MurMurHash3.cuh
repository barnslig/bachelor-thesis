/**
 * @file
 * @brief The MurMur3 128-bit hash, with CUDA support
 * @see https://github.com/aappleby/smhasher
 */
#ifndef MURMUR_HASH_3_CUH_
#define MURMUR_HASH_3_CUH_

#include <cstdint>

/**
 * Create a 128-bit hash
 *
 * Actually MurmurHash3_x64_128
 *
 * @example
 *   char *elem = "aaa123";
 *   uint64_t hash[2];
 *   MurmurHash3_128(elem, sizeof(elem), 0, &hash);
 *   // hash[0] and hash[1] contain the hash
 *
 * @param key The pointer to the value to be hashed
 * @param len The length of the key
 * @param seed Any 32-bit value
 * @param out The pointer to the output array
 */
__host__ __device__ void MurmurHash3_128(const void *key, int len, uint32_t seed, void *out);

#endif // MURMUR_HASH_3_CUH_

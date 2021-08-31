/**
 * @file
 * @brief HyperLogLog cardinality estimator
 * @see https://github.com/hideo55/cpp-HyperLogLog
 * @see https://github.com/armon/hlld
 * @see http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
 */
#ifndef HYPERLOGLOG_HPP_
#define HYPERLOGLOG_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "HyperLogLog_constants.cuh"
#include "MurMurHash3.cuh"

__host__ __device__ inline int myClzll(long long int x)
{
#ifdef __CUDA_ARCH__
  return __clzll(x);
#else // just hope that we are in g++/clang
  return __builtin_clzll(x);
#endif
}

/**
 * Implement of 'HyperLogLog' estimate cardinality algorithm
 *
 * @tparam T The type of the elements counted
 * @tparam Tsize The size of T, i.e. sizeof(T)
 * @tparam B The precision / register bit width (register size will be 2 to the B power). This value must be in the range [4, 30].
 */
template <typename T, size_t Tsize, uint8_t B>
class HyperLogLog
{
  protected:
  /**
   * Register size
   */
  uint32_t m_ = 1 << B;

  /**
   * alpha * m^2
   */
  double alphaMM_;

  /**
   * Registers
   */
  uint32_t M_[1 << B] = {};

  /**
   * Computes the raw cardinality estimate
   */
  double raw_estimate(unsigned int *num_zero)
  {
    double sum = 0;
    for (int i = 0; i < m_; i++)
    {
      sum += 1.0 / (1 << M_[i]);
      if (M_[i] == 0)
      {
        *num_zero += 1;
      }
    }
    return alphaMM_ / sum;
  }

  /**
   * Binary searches for the nearest matching index
   * @return The matching index, or closest match
   */
  static int binary_search(double val, int num, const double *array)
  {
    int low = 0, mid, high = num - 1;
    while (low < high)
    {
      mid = (low + high) / 2;
      if (val > array[mid])
      {
        low = mid + 1;
      }
      else if (val == array[mid])
      {
        return mid;
      }
      else
      {
        high = mid - 1;
      }
    }
    return low;
  }

  /**
   * Interpolates the bias estimate using the
   * empircal data collected by Google, from the
   * paper mentioned above.
   */
  double bias_estimate(double raw_est)
  {
    // Determine the samples available
    int samples;
    switch (B)
    {
    case 4:
      samples = 80;
      break;
    case 5:
      samples = 160;
      break;
    default:
      samples = 200;
      break;
    }

    // Get the proper arrays based on precision
    double *estimates = *(rawEstimateData + (B - 4));
    double *biases = *(biasData + (B - 4));

    // Get the matching biases
    int idx = binary_search(raw_est, samples, estimates);
    if (idx == 0)
    {
      return biases[0];
    }
    else if (idx == samples)
    {
      return biases[samples - 1];
    }
    else
    {
      return (biases[idx] + biases[idx - 1]) / 2;
    }
  }

  /**
   * Estimates cardinality using a linear counting.
   * Used when some registers still have a zero value.
   */
  double linear_count(unsigned int num_zero)
  {
    return m_ * log((double)m_ / (double)num_zero);
  }

  public:
  __host__ __device__ HyperLogLog()
  {
    double alpha;
    switch (m_)
    {
    case 16:
      alpha = 0.673;
      break;
    case 32:
      alpha = 0.697;
      break;
    case 64:
      alpha = 0.709;
      break;
    default:
      alpha = 0.7213 / (1.0 + 1.079 / m_);
      break;
    }
    alphaMM_ = alpha * m_ * m_;
  }

  /**
   * Adds element to the estimator
   *
   * @param str string to add
   * @param len length of string
   */
  __host__ __device__ void add(T *elem)
  {
    uint64_t hash[2];
    MurmurHash3_128(elem, Tsize, 0, &hash);

    // Truncate the MurMur3 hash to 32-bit
    // uint32_t my_hash = hash[1] & (((uint64_t)1 << 32) - 1);

    // First B bytes are the index
    uint64_t index = hash[1] >> (64 - B);

    // Count the number of leading zeros of the remaining 64 - B bits
    uint8_t rank = myClzll(hash[1] << B) + 1;

    if (rank > M_[index])
    {
      M_[index] = rank;
    }
  }

  /**
   * Estimates cardinality value.
   *
   * @returns Estimated cardinality value.
   */
  double estimate()
  {
    unsigned int num_zero = 0;
    double raw_est = raw_estimate(&num_zero);

    // Check if we need to apply bias correction
    if (raw_est <= 5 * m_)
    {
      raw_est -= bias_estimate(raw_est);
    }

    // Check if linear counting should be used
    double alt_est;
    if (num_zero)
    {
      alt_est = linear_count(num_zero);
    }
    else
    {
      alt_est = raw_est;
    }

    // Determine which estimate to use
    if (alt_est <= switchThreshold[B - 4])
    {
      return alt_est;
    }
    else
    {
      return raw_est;
    }
  }

  /**
   * Merges the estimate from 'other' into this object, returning the estimate of their union.
   * The number of registers in each must be the same.
   *
   * @param other HyperLogLog instance to be merged
   */
  __host__ void merge(const HyperLogLog<T, Tsize, B> &other)
  {
    for (unsigned int r = 0; r < m_; ++r)
    {
      if (M_[r] < other.M_[r])
      {
        M_[r] |= other.M_[r];
      }
    }
  }

  /**
   * Returns the standard error
   *
   * From the paper: The estimates provided by HyperLogLog are expected
   * to be within σ, 2σ, 3σ of the exact count in respectively 65%,
   * 95%, 99% of all the cases. σ is the standard error.
   *
   * @returns Standard error
   */
  double error() const
  {
    return 1.04 / sqrt(m_);
  }
};

#endif // HYPERLOGLOG_HPP_

/**
 * @file
 * @brief HyperLogLog cardinality estimator
 * @see https://github.com/hideo55/cpp-HyperLogLog
 * @see http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
 */
#ifndef HYPERLOGLOG_HPP_
#define HYPERLOGLOG_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "JenkinsHash.cuh"

constexpr unsigned int kHllHashSeed = 313;

constexpr double kPow_2_32 = 4294967296.0;     ///< 2^32
constexpr double kNegPow_2_32 = -4294967296.0; ///< -(2^32)

__host__ __device__ inline int myClz(int x)
{
#ifdef __CUDA_ARCH__
  return __clz(x);
#else // just hope that we are in g++/clang
  return __builtin_clz(x);
#endif
}

/**
 * Implement of 'HyperLogLog' estimate cardinality algorithm
 *
 * @tparam T The type of the elements counted
 * @tparam Tsize The size of T, i.e. sizeof(T)
 * @tparam B The register bit width (register size will be 2 to the B power). This value must be in the range [4, 30].
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
  uint8_t M_[1 << B] = {};

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
    /* The ~magic values~ are the golden ratio from the original Jenkins
     * hash which we have removed in our implementation. Here, we just
     * re-apply them so the algorithm works again as originially
     * intended.
     *
     * We use the Jenkins hash instead of the commonly used MurmurHash3
     * solely by the reason that it is already part of our codebase.
     */
    uint32_t hash = jenkins_hash(reinterpret_cast<uint8_t *>(elem), Tsize, 0x9e3779b9, 0x9e3779b9, kHllHashSeed);

    uint32_t index = hash >> (32 - B);
    uint8_t rank = min(32 - B, myClz(hash << B)) + 1;
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
  __host__ __device__ double estimate() const
  {
    double estimate;
    double sum = 0.0;
    for (uint32_t i = 0; i < m_; i++)
    {
      sum += 1.0 / (1 << M_[i]);
    }
    estimate = alphaMM_ / sum; // E in the original paper
    if (estimate <= 2.5 * m_)
    {
      uint32_t zeros = 0;
      for (uint32_t i = 0; i < m_; i++)
      {
        if (M_[i] == 0)
        {
          zeros++;
        }
      }
      if (zeros != 0)
      {
        estimate = m_ * log(static_cast<double>(m_) / zeros);
      }
    }
    else if (estimate > (1.0 / 30.0) * kPow_2_32)
    {
      estimate = kNegPow_2_32 * log(1.0 - (estimate / kPow_2_32));
    }
    return estimate;
  }

  /**
   * Merges the estimate from 'other' into this object, returning the estimate of their union.
   * The number of registers in each must be the same.
   *
   * @param other HyperLogLog instance to be merged
   */
  __host__ __device__ void merge(const HyperLogLog<T, Tsize, B> &other)
  {
    for (uint32_t r = 0; r < m_; ++r)
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
  __host__ __device__ double error() const
  {
    return 1.04 / sqrt(m_);
  }
};

#endif // HYPERLOGLOG_HPP_

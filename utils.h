#ifndef UTIL_HEADER_CUH
#define UTIL_HEADER_CUH

#include "aabb.h"
#include "helper_math.h"
#include <iostream>
#include <string>

#define CHECK_CUDA(FN)                                                         \
  {                                                                            \
    auto fn_err = FN;                                                          \
    if (fn_err != cudaSuccess) {                                               \
      std::cout << #FN << " failed due to " << cudaGetErrorName(fn_err)        \
                << ": " << cudaGetErrorString(fn_err) << std::endl             \
                << std::flush;                                                 \
      throw std::runtime_error(#FN);                                           \
    }                                                                          \
  }

__device__ __host__ inline unsigned int expand_bits(unsigned int v) noexcept {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ __host__ inline unsigned int
morton_code(float4 xyz, float resolution = 1024.0f) noexcept {
  xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
  xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
  xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
  const unsigned int xx = expand_bits(static_cast<unsigned int>(xyz.x));
  const unsigned int yy = expand_bits(static_cast<unsigned int>(xyz.y));
  const unsigned int zz = expand_bits(static_cast<unsigned int>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

__device__ __host__ inline unsigned int
morton_code(double4 xyz, double resolution = 1024.0) noexcept {
  xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
  xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
  xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
  const unsigned int xx = expand_bits(static_cast<unsigned int>(xyz.x));
  const unsigned int yy = expand_bits(static_cast<unsigned int>(xyz.y));
  const unsigned int zz = expand_bits(static_cast<unsigned int>(xyz.z));
  return xx * 4 + yy * 2 + zz;
}

__device__ inline aabb aabb_merge(const aabb &lhs, const aabb &rhs) noexcept {
  return aabb{make_float4(::fmaxf(lhs.upper.x, rhs.upper.x),
                          ::fmaxf(lhs.upper.y, rhs.upper.y),
                          ::fmaxf(lhs.upper.z, rhs.upper.z), 0.0f),
              make_float4(::fminf(lhs.lower.x, rhs.lower.x),
                          ::fminf(lhs.lower.y, rhs.lower.y),
                          ::fminf(lhs.lower.z, rhs.lower.z), 0.0f)};
}

__device__ inline float4 aabb_center(const aabb &box) noexcept {
  float4 center;
  center.x = (box.lower.x + box.upper.x) * 0.5f;
  center.y = (box.lower.y + box.upper.y) * 0.5f;
  center.z = (box.lower.z + box.upper.z) * 0.5f;
  return center;
}

class Timer {
public:
  Timer(std::string name) : m_name(name) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
  }

  ~Timer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << m_name + " time: " << elapsedTime << "ms" << std::endl;
  }

private:
  std::string m_name;
  cudaEvent_t start, stop;
};

#endif

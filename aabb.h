#ifndef AABB_HEADER_H
#define AABB_HEADER_H

#include "helper_math.h"
#include <cuda_runtime.h>

struct aabb {
  float4 upper;
  float4 lower;
};

__device__ inline aabb aabb_merge(const aabb &lhs, const aabb &rhs) noexcept;

__device__ inline float4 aabb_center(const aabb &box) noexcept;

#endif

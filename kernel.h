#pragma once

#include "aabb.h"
#include "bvh.h"
#include "helper_math.h"
#include "utils.h"

struct triangle_mesh_aabb {
  triangle_mesh_aabb() = default;
  triangle_mesh_aabb(const float4 *buffer) : m_buffer(buffer) {}
  __device__ aabb operator()(int3 idx) const noexcept {
    aabb box;
    const float4 &v0 = m_buffer[idx.x];
    const float4 &v1 = m_buffer[idx.y];
    const float4 &v2 = m_buffer[idx.z];

    box.lower.x = fminf(fminf(v0.x, v1.x), v2.x);
    box.lower.y = fminf(fminf(v0.y, v1.y), v2.y);
    box.lower.z = fminf(fminf(v0.z, v1.z), v2.z);

    box.upper.x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    box.upper.y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    box.upper.z = fmaxf(fmaxf(v0.z, v1.z), v2.z);

    return box;
  };
  const float4 *m_buffer;
};

struct default_morton_code_calculator {
  default_morton_code_calculator(aabb w) : whole(w) {}
  default_morton_code_calculator() = default;
  ~default_morton_code_calculator() = default;
  default_morton_code_calculator(default_morton_code_calculator const &) =
      default;
  default_morton_code_calculator(default_morton_code_calculator &&) = default;
  default_morton_code_calculator &
  operator=(default_morton_code_calculator const &) = default;
  default_morton_code_calculator &
  operator=(default_morton_code_calculator &&) = default;

  __device__ unsigned int operator()(const aabb &box) noexcept {
    auto p = aabb_center(box);
    p.x -= whole.lower.x;
    p.y -= whole.lower.y;
    p.z -= whole.lower.z;
    p.x /= (whole.upper.x - whole.lower.x);
    p.y /= (whole.upper.y - whole.lower.y);
    p.z /= (whole.upper.z - whole.lower.z);
    return morton_code(p);
  }
  aabb whole;
};

template class bvh<int3, float4, triangle_mesh_aabb,
                   default_morton_code_calculator>;

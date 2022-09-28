#pragma once

#include <cuda_runtime.h>

/**
 * @brief A simple camera class
 *
 */
struct camera {
  float4 pos;      // Camera position
  float4 view;     // View direction
  float4 up;       // Up direction
  float2 fov;      // Field of view
  float aperture;  // Aperture
  float focalDist; // Focal distance
};

#include "ray.h"
/**
 * @brief Get the Ray object
 *
 * @param cam the camera, which is used to generate the ray
 * @param uv the uv coordinates, where (0,0) is the bottom left corner and (1,1)
 * @param jitter the jitter, which is used to create a depth of field effect
 * @return ray
 */
__device__ __host__ ray getCamRay(const camera &cam, float2 uv, float2 jitter) {
  float3 pos = make_float3(cam.pos);
  float3 view = make_float3(cam.view);
  float3 up = make_float3(cam.up);

  float3 right = normalize(cross(view, up));
  float3 up2 = normalize(cross(right, view));

  float3 dir = normalize(view + right * (uv.x - 0.5f) * cam.fov.x +
                         up2 * (uv.y - 0.5f) * cam.fov.y);

  float3 jitteredPos =
      pos + right * jitter.x * cam.aperture + up2 * jitter.y * cam.aperture;

  return ray{
      make_float4(jitteredPos, 0.0f),
      make_float4(normalize(jitteredPos + dir * cam.focalDist - pos), 0.0f)};
}

#pragma once

#include <cuda_runtime.h>

/**
 * @brief A simple camera class
 *
 */
struct camera
{
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
__device__ __host__ ray getCamRay(const camera &cam, float2 uv, float2 jitter)
{
  float4 origin = cam.pos;
  float4 direction = cam.view;
  float4 up = cam.up;
  float4 right = normalize(cross(cam.view, cam.up));

  float2 fov = cam.fov;
  float aperture = cam.aperture;
  float focalDist = cam.focalDist;

  float2 jittered = (jitter - 0.5f) * aperture;
  origin += right * jittered.x;
  origin += up * jittered.y;

  direction += right * (uv.x - 0.5f) * fov.x;
  direction += up * (uv.y - 0.5f) * fov.y;
  direction = normalize(direction) * focalDist;

  return ray{origin, direction};
}
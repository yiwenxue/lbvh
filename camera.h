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

#pragma once

#include <cuda_runtime.h>

struct camera {
  float4 pos;
  float4 view;
  float4 up;
  float2 fov;
  float aperture;
  float focalDist;
};
